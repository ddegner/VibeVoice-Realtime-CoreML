import Foundation
import CoreML

/// Autoregressive TTS generation loop with voice-prompt conditioning.
///
/// The voice prompt `.npy` caches (from `08_export_voice_prompt.py`) provide
/// pre-computed KV state for all four branches (lm, tts_lm, neg_lm, neg_tts_lm).
/// New text tokens are fed through the **decode** models (not prefill),
/// extending the cache incrementally — matching upstream's interleaved
/// text-window / speech-window architecture.
///
/// Architecture (matching `modeling_vibevoice_streaming_inference.py:generate`):
///   1. Load voice prompt → initial KV caches for all branches.
///   2. Feed text tokens in windows of 5 through text-LM decode, then
///      forward TTS-LM decode with `lm_last_hidden_state` = text-LM output
///      and `tts_text_masks = 1` (text position).
///   3. After each text window, sample up to 6 speech frames:
///      - Diffusion per frame with CFG (positive + negative branches).
///      - `acoustic_connector(latent)` fed back as next `lm_hidden`.
///      - TTS-LM decode on both branches with `tts_text_masks = 0`.
///      - EOS check on positive branch.
///   4. Repeat 2–3 until all text consumed, then continue speech-only
///      until EOS fires.
///   5. VAE decode in 16-frame windows.
public final class TTSLoop: @unchecked Sendable {
    public struct Config {
        public var maxFrames: Int = 200       // ~26.6s ceiling
        public var eosThreshold: Float = 0.5
        public var textWindowSize: Int = 5
        public var speechWindowSize: Int = 6
        /// Latent frames of left-context the VAE is fed at each sliding-window
        /// decode. Because the acoustic decoder is fully **causal** with a
        /// stem receptive field of 6 latent frames (kernel=7), L≥7 makes the
        /// output for the "new" portion numerically identical to the upstream
        /// streaming path. L=8 gives a comfortable margin and a clean 50/50
        /// split with T=16, at the cost of 2× VAE calls (~0.14s each on GPU).
        public var vaeLeftContext: Int = 8
        /// Diagnostic: print one line per speech frame with the EOS classifier
        /// probability. Use to investigate clip-boundary artifacts (a sharp
        /// rise from ~0 to ~0.8 in one frame suggests the last frame was a
        /// phoneme onset; a slow creep suggests threshold tuning instead).
        public var logEOSTrajectory: Bool = false
        /// Post-generation: trim leading frames whose RMS is below
        /// `silenceFloor`. VibeVoice often emits 0.5–1s of silence before
        /// speaking when the initial diffusion noise doesn't align with a
        /// phoneme onset. This removes that dead air. Disabled -> keep raw.
        public var trimHeadSilence: Bool = true
        /// Post-generation: attempt to detect a trailing ghost-phoneme
        /// (a word onset the EOS classifier caught too late). The signal
        /// is a valley-then-bump in the RMS trajectory within the last
        /// 4 frames. **Disabled by default** because this signature is
        /// ambiguous: it also matches a natural plosive release after a
        /// vowel (e.g., the 'g' in "dog"), so trimming can cut real
        /// speech. Enable only for voices/texts you've validated by ear.
        public var trimTailOnset: Bool = false
        /// RMS below which a frame is considered silence for head trimming.
        public var silenceFloor: Float = 0.01
        public init() {}
    }

    private let llm: LLMRunner
    private let diffusion: DiffusionRunner
    private let vae: VAERunner
    private let heads: HeadsRunner

    public init(llm: LLMRunner, diffusion: DiffusionRunner, vae: VAERunner, heads: HeadsRunner) {
        self.llm = llm
        self.diffusion = diffusion
        self.vae = vae
        self.heads = heads
    }

    /// Per-stage timing breakdown from the last `generate` call.
    public struct Timing: CustomStringConvertible {
        public var textLMDecode: Double = 0
        public var ttsLMDecode: Double = 0
        public var diffusion: Double = 0
        public var connector: Double = 0
        public var eos: Double = 0
        public var vaeDecode: Double = 0
        public var total: Double = 0
        public var speechFrames: Int = 0
        /// Number of DPM-Solver++ denoising steps per speech frame (each step
        /// is 2 CFG passes through the diffusion head). Populated by `generate`.
        public var diffusionSteps: Int = 0

        public var description: String {
            let perFrame = speechFrames > 0 ? total / Double(speechFrames) : 0
            return """
            Timing (\(speechFrames) speech frames, \(String(format: "%.2f", total))s total, \(String(format: "%.0f", perFrame * 1000))ms/frame):
              text_lm_decode:  \(String(format: "%6.2f", textLMDecode))s
              tts_lm_decode:   \(String(format: "%6.2f", ttsLMDecode))s  (pos + neg)
              diffusion:       \(String(format: "%6.2f", diffusion))s  (\(diffusionSteps) steps × 2 CFG)
              connector:       \(String(format: "%6.2f", connector))s
              eos_classifier:  \(String(format: "%6.2f", eos))s
              vae_decode:      \(String(format: "%6.2f", vaeDecode))s
            """
        }
    }

    /// Last timing breakdown. Read after `generate` returns.
    public private(set) var lastTiming = Timing()

    /// Run the loop with a voice prompt. Returns concatenated 24 kHz Float32 PCM.
    public func generate(
        tokens: [Int32],
        voicePrompt: VoicePrompt,
        config: Config = Config()
    ) async throws -> [Float] {
        var timing = Timing()
        let wallStart = CFAbsoluteTimeGetCurrent()

        // ---- State from voice prompt ----
        var lmK = voicePrompt.lm.kCache
        var lmV = voicePrompt.lm.vCache
        var lmPos = Int32(voicePrompt.lm.length)
        // Note: the voice prompt's last LM hidden state is intentionally NOT
        // used here. Upstream's TTS-LM consumes the per-token hidden state
        // produced by text_lm_decode (stored in `perTokenHidden` below), not
        // the prefill tail — so we never need a persistent `lmHidden` var.

        var posK = voicePrompt.ttsLM.kCache
        var posV = voicePrompt.ttsLM.vCache
        var posPos = Int32(voicePrompt.ttsLM.length)

        var negK = voicePrompt.negTTSLM.kCache
        var negV = voicePrompt.negTTSLM.vCache
        var negPos = Int32(voicePrompt.negTTSLM.length)

        // Last hidden from TTS LM voice prompt
        var posLast = try lastPosition(of: voicePrompt.ttsLM.hidden)
        var negLast = try lastPosition(of: voicePrompt.negTTSLM.hidden)

        var audio: [Float] = []
        // Full history of generated latent frames. We slide a 16-frame VAE
        // window forward over this so each decode sees valid left context
        // instead of zeros at every window boundary (which caused the
        // "jumble before 'before'" artifact at speech frame ~16).
        var allLatents: [MLMultiArray] = []
        var decodedFrames = 0
        let stride = max(1, DiffusionRunner.T - config.vaeLeftContext)
        var totalSpeechFrames = 0
        var finished = false

        // ---- Interleaved text/speech windows ----
        var textOffset = 0

        while !finished {
            // == Text window: feed up to `textWindowSize` tokens ==
            let windowEnd = min(textOffset + config.textWindowSize, tokens.count)
            let textWindow = Array(tokens[textOffset..<windowEnd])
            textOffset = windowEnd

            if !textWindow.isEmpty {
                // Feed each text token through text-LM decode, collecting
                // per-token hidden states so we can pair them 1:1 with the
                // TTS-LM steps (upstream feeds the full window's hidden as
                // `lm_last_hidden_state` — we replicate in decode mode).
                var perTokenHidden: [MLMultiArray] = []
                for tok in textWindow {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let step = try await llm.textDecodeStep(
                        inputID: tok, kCache: lmK, vCache: lmV, positionID: lmPos)
                    timing.textLMDecode += CFAbsoluteTimeGetCurrent() - t0
                    lmK = step.k; lmV = step.v
                    perTokenHidden.append(step.hidden)
                    lmPos += 1
                }

                for (i, tok) in textWindow.enumerated() {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let posStep = try await llm.ttsDecodeStep(
                        inputID: tok, lmHidden: perTokenHidden[i],
                        tailMask: 1.0, textMask: 1,
                        kCache: posK, vCache: posV, positionID: posPos)
                    timing.ttsLMDecode += CFAbsoluteTimeGetCurrent() - t0
                    posK = posStep.k; posV = posStep.v
                    posLast = try lastPosition(of: posStep.hidden)
                    posPos += 1
                }
            }

            // == Speech window: sample up to `speechWindowSize` frames ==
            for _ in 0..<config.speechWindowSize {
                if totalSpeechFrames >= config.maxFrames { finished = true; break }

                // Diffusion sample one frame using CFG
                let tDiff = CFAbsoluteTimeGetCurrent()
                let latent = try await diffusion.denoiseFrame(cond: posLast, uncond: negLast)
                timing.diffusion += CFAbsoluteTimeGetCurrent() - tDiff
                allLatents.append(latent)
                totalSpeechFrames += 1

                // Sliding-window VAE decode.
                //  - First window: fires when we have the first T=16 frames;
                //    emits all 16 frames of audio (no left context available).
                //  - Subsequent windows: fire every `stride` new frames; the
                //    window is `[newEnd - T .. newEnd - 1]`, and we discard
                //    the first `L*3200` audio samples (their latent frames
                //    were already emitted by the previous decode).
                while shouldDecodeNext(decoded: decodedFrames,
                                       total: allLatents.count,
                                       stride: stride) {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try await decodeSliding(
                        allLatents: allLatents,
                        leftContext: config.vaeLeftContext,
                        decodedFrames: &decodedFrames,
                        appendTo: &audio)
                    timing.vaeDecode += CFAbsoluteTimeGetCurrent() - t0
                }

                // Project latent → LM hidden for feedback
                let tConn = CFAbsoluteTimeGetCurrent()
                let acousticEmbed = try await heads.project(latent: latent)
                timing.connector += CFAbsoluteTimeGetCurrent() - tConn

                // Advance both TTS LM branches (speech position).
                // pos and neg share nothing but `acousticEmbed` (read-only)
                // and the `llm` model reference, so we fire both predictions
                // concurrently — the MLModel backend is thread-safe and
                // LLMRunner now allocates per-call scratch buffers instead
                // of sharing stored properties.
                //
                // The `let` bindings below are required for Swift 6 strict
                // concurrency: `async let` must capture local `let`s (or
                // `Sendable` values) rather than mutable `var`s, so we snapshot
                // the K/V caches and positions before spawning the tasks.
                let tTts = CFAbsoluteTimeGetCurrent()
                let posKIn = posK, posVIn = posV, posPosIn = posPos
                let negKIn = negK, negVIn = negV, negPosIn = negPos
                async let posTask = llm.ttsDecodeStep(
                    inputID: 1, lmHidden: acousticEmbed,
                    tailMask: 1.0, textMask: 0,
                    kCache: posKIn, vCache: posVIn, positionID: posPosIn)
                async let negTask = llm.ttsDecodeStep(
                    inputID: 1, lmHidden: acousticEmbed,
                    tailMask: 1.0, textMask: 0,
                    kCache: negKIn, vCache: negVIn, positionID: negPosIn)
                let (posStep, negStep) = try await (posTask, negTask)
                posK = posStep.k; posV = posStep.v
                posLast = try lastPosition(of: posStep.hidden)
                posPos += 1
                negK = negStep.k; negV = negStep.v
                negLast = try lastPosition(of: negStep.hidden)
                negPos += 1
                timing.ttsLMDecode += CFAbsoluteTimeGetCurrent() - tTts

                // EOS check
                let tEos = CFAbsoluteTimeGetCurrent()
                let eosProb = try await heads.eosProbability(hidden: posLast)
                timing.eos += CFAbsoluteTimeGetCurrent() - tEos
                if config.logEOSTrajectory {
                    let fired = eosProb > config.eosThreshold ? " <-- EOS" : ""
                    print(String(format: "  frame %3d: eos_prob = %.4f%@",
                                 totalSpeechFrames - 1, eosProb, fired))
                }
                if eosProb > config.eosThreshold {
                    finished = true; break
                }
            }

            // If no more text and speech didn't finish, continue speech-only
            if textOffset >= tokens.count && !finished {
                // Keep generating speech frames until EOS or max
                continue  // loops back, textWindow will be empty
            }
        }

        // Final drain: emit audio for any latent frames that didn't reach a
        // stride boundary. Two cases:
        //   (a) Fewer than T frames total ever generated — same zero-padded
        //       single-window path as before.
        //   (b) Some "tail" frames past the last stride boundary — decode a
        //       window of the LAST T frames (so the tail sees full left
        //       context), and discard everything already emitted.
        if decodedFrames < allLatents.count {
            let t0 = CFAbsoluteTimeGetCurrent()
            try await drainRemaining(
                allLatents: allLatents,
                decodedFrames: &decodedFrames,
                appendTo: &audio)
            timing.vaeDecode += CFAbsoluteTimeGetCurrent() - t0
        }

        // Post-generation boundary cleanup. Two independent artifacts:
        //  1. Head silence: the model often needs several frames for the
        //     diffusion noise to resolve into a phoneme, producing a long
        //     silent lead-in that sounds like a delay.
        //  2. Tail ghost phoneme: the EOS classifier is a lagging indicator.
        //     When it fires with a sharp jump (prev ~0.01 → curr > 0.5), the
        //     last 1–3 frames contain a truncated onset of a word the model
        //     decided not to complete. This is also present upstream — same
        //     architecture, same bug. We detect and remove it via the RMS
        //     trajectory, since the signature (valley + rising edge within
        //     the last 4 frames) is reliable and only fires on affected clips.
        audio = trimBoundaries(audio, config: config)

        timing.speechFrames = totalSpeechFrames
        timing.diffusionSteps = diffusion.steps
        timing.total = CFAbsoluteTimeGetCurrent() - wallStart
        lastTiming = timing
        return audio
    }

    // MARK: - Boundary trim

    /// Apply head-silence and tail-onset trimming to the generated audio.
    /// Pure function of the audio buffer — does not change any model state.
    /// Short (≤10ms) linear fades are applied at each trim point to avoid
    /// DC clicks from truncating mid-cycle.
    private func trimBoundaries(_ audio: [Float], config: Config) -> [Float] {
        guard !audio.isEmpty else { return audio }
        let fs = Self.samplesPerFrame
        let n = audio.count / fs
        guard n >= 2 else { return audio }

        // Per-frame RMS (cheap — n is at most a couple hundred).
        var rms = [Float](repeating: 0, count: n)
        audio.withUnsafeBufferPointer { buf in
            for i in 0..<n {
                var sumSq: Double = 0
                let base = i * fs
                for j in 0..<fs {
                    let x = Double(buf[base + j])
                    sumSq += x * x
                }
                rms[i] = Float((sumSq / Double(fs)).squareRoot())
            }
        }

        var startFrame = 0
        var endFrame = n

        if config.trimHeadSilence {
            // Walk forward while below silence floor; always leave at least
            // 1 frame so we don't eliminate a short utterance entirely.
            while startFrame < n - 1 && rms[startFrame] < config.silenceFloor {
                startFrame += 1
            }
        }

        if config.trimTailOnset && n >= 5 {
            // Scan the last 4 frames (excluding the very last, which can't be
            // a local minimum by our definition) for a valley-then-bump.
            let K = 4
            for i in max(startFrame + 1, n - K)..<(n - 1) {
                let isLocalMin = rms[i] < rms[i-1] && rms[i] < rms[i+1]
                if !isLocalMin { continue }
                // Any subsequent frame significantly louder than the valley?
                var maxAfter: Float = 0
                for k in (i + 1)..<n { maxAfter = max(maxAfter, rms[k]) }
                if maxAfter > 1.5 * rms[i] {
                    endFrame = i + 1   // keep up through frame i
                    break
                }
            }
        }

        guard startFrame > 0 || endFrame < n else { return audio }

        var trimmed = Array(audio[(startFrame * fs)..<(endFrame * fs)])
        // Short linear fades at the cut edges (10ms @ 24 kHz = 240 samples).
        // Guard against degenerate clips shorter than 2× the fade length.
        let fadeLen = min(240, trimmed.count / 2)
        if startFrame > 0 && fadeLen > 0 {
            for i in 0..<fadeLen { trimmed[i] *= Float(i) / Float(fadeLen) }
        }
        if endFrame < n && fadeLen > 0 {
            let off = trimmed.count - fadeLen
            for i in 0..<fadeLen {
                trimmed[off + i] *= Float(fadeLen - i) / Float(fadeLen)
            }
        }
        return trimmed
    }

    // MARK: - helpers

    private func lastPosition(of hidden: MLMultiArray) throws -> MLMultiArray {
        let shape = hidden.shape.map(\.intValue)
        precondition(shape.count == 3 && shape[0] == 1 && shape[2] == 896,
                     "unexpected hidden shape \(shape)")
        let T = shape[1]
        let out = try MLMultiArray(shape: [1, 896], dataType: .float32)
        let src = hidden.dataPointer.assumingMemoryBound(to: Float.self)
        let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(dst, src.advanced(by: (T - 1) * 896), 896 * MemoryLayout<Float>.size)
        return out
    }

    // MARK: - VAE sliding-window decode

    private static let samplesPerFrame = 3200  // 24 kHz / (8*5*5*4*2*2)

    /// Decide whether we have enough buffered latents to fire another decode.
    /// - First decode (decodedFrames == 0): need a full window (T frames).
    /// - Subsequent: need `stride` more frames past what we've already emitted.
    private func shouldDecodeNext(decoded: Int, total: Int, stride: Int) -> Bool {
        if decoded == 0 {
            return total >= DiffusionRunner.T
        }
        // Next window starts at `decoded - L` and ends at `decoded - L + T - 1`
        // = `decoded + stride - 1`. So we need `total > decoded + stride - 1`.
        return total >= decoded + stride
    }

    /// Decode one sliding window and append the newly-valid samples to `audio`.
    /// Advances `decodedFrames` by the number of frames newly emitted.
    ///
    /// Layout on the i-th call (0-indexed):
    ///   i=0: windowStart=0,        framesToEmit=T           (no left ctx)
    ///   i>0: windowStart=dec-L,    framesToEmit=stride      (L frames of
    ///                                                         left context)
    @discardableResult
    private func decodeSliding(
        allLatents: [MLMultiArray],
        leftContext L: Int,
        decodedFrames: inout Int,
        appendTo audio: inout [Float]
    ) async throws -> Int {
        let T = DiffusionRunner.T
        let windowStart: Int
        let framesToEmit: Int
        if decodedFrames == 0 {
            windowStart = 0
            framesToEmit = T
        } else {
            windowStart = decodedFrames - L
            framesToEmit = T - L
        }
        return try await decodeWindow(
            allLatents: allLatents,
            windowStart: windowStart,
            framesToEmit: framesToEmit,
            decodedFrames: &decodedFrames,
            appendTo: &audio)
    }

    /// Core of sliding-window decode.
    ///   - `windowStart`: first latent index fed to the VAE.
    ///   - `framesToEmit`: how many contiguous frames' audio to append.
    ///
    /// We emit audio for latent indices `[decodedFrames ..
    /// decodedFrames + framesToEmit - 1]`. Those correspond to window offsets
    /// `[decodedFrames - windowStart ..]` — i.e., we skip
    /// `(decodedFrames - windowStart) * 3200` samples at the front of the
    /// VAE output, keep the next `framesToEmit * 3200`, and discard the rest
    /// (either already-emitted left context, or zero-padded right context).
    /// Because the decoder is fully causal, output samples for an index `t`
    /// depend only on inputs `[0..t]` within the window's receptive field,
    /// so zero-right-padding (for short-utterance drain) is safe.
    private func decodeWindow(
        allLatents: [MLMultiArray],
        windowStart: Int,
        framesToEmit: Int,
        decodedFrames: inout Int,
        appendTo audio: inout [Float]
    ) async throws -> Int {
        let T = DiffusionRunner.T
        precondition(framesToEmit > 0 && framesToEmit <= T,
                     "framesToEmit out of range: \(framesToEmit)")
        precondition(windowStart >= 0, "windowStart negative")
        precondition(windowStart <= decodedFrames,
                     "windowStart \(windowStart) > decodedFrames \(decodedFrames) — would leave a gap")
        precondition(decodedFrames + framesToEmit <= windowStart + T,
                     "emit range [\(decodedFrames), \(decodedFrames + framesToEmit)) exceeds window [\(windowStart), \(windowStart + T))")

        let out = try MLMultiArray(shape: [1, 64, NSNumber(value: T)], dataType: .float32)
        let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
        let scale = heads.constants.speech_scaling_factor
        let bias = heads.constants.speech_bias_factor

        // Fill window. If the buffer is shorter than windowStart+T (only
        // possible on the short-utterance drain), zero-pad the tail. Causal
        // convs mean this doesn't affect output for earlier positions.
        let available = min(T, max(0, allLatents.count - windowStart))
        for t in 0..<available {
            let frame = allLatents[windowStart + t]
            let src = frame.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<64 { dst[c * T + t] = src[c] / scale - bias }
        }
        if available < T {
            for t in available..<T {
                for c in 0..<64 { dst[c * T + t] = 0 }
            }
        }

        let samples = try await vae.decode(latents: out)
        let skipFrames = decodedFrames - windowStart
        let skip = skipFrames * Self.samplesPerFrame
        let keep = framesToEmit * Self.samplesPerFrame
        precondition(samples.count >= skip + keep,
                     "VAE output \(samples.count) < skip+keep \(skip + keep)")
        audio.append(contentsOf: samples[skip..<(skip + keep)])
        decodedFrames += framesToEmit
        return framesToEmit
    }

    /// Drain whatever latent frames haven't been emitted yet. Picks the
    /// window that gives the tail the most left context, i.e. aligned to
    /// the end of `allLatents` so the new audio sees the full T=16 history.
    private func drainRemaining(
        allLatents: [MLMultiArray],
        decodedFrames: inout Int,
        appendTo audio: inout [Float]
    ) async throws {
        let T = DiffusionRunner.T
        let totalFrames = allLatents.count
        let remaining = totalFrames - decodedFrames
        precondition(remaining > 0, "drain called with nothing to emit")

        if totalFrames < T {
            // Only ever generated a tiny amount — single zero-padded decode,
            // keep only the valid prefix (same behaviour as the old drain).
            _ = try await decodeWindow(
                allLatents: allLatents,
                windowStart: 0,
                framesToEmit: remaining,
                decodedFrames: &decodedFrames,
                appendTo: &audio)
            return
        }

        // Slide the window so it ends at totalFrames - 1. That gives the
        // tail up to T-remaining frames of valid left context.
        let windowStart = totalFrames - T
        _ = try await decodeWindow(
            allLatents: allLatents,
            windowStart: windowStart,
            framesToEmit: remaining,
            decodedFrames: &decodedFrames,
            appendTo: &audio)
    }
}
