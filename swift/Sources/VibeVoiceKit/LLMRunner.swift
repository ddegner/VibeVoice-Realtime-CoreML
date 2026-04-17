import Foundation
import CoreML

/// Drives the text LM (prefill) and — eventually — the TTS LM autoregressive
/// decode loop. For now this is a prefill-only harness so the rest of the
/// pipeline (diffusion, VAE) can be wired end-to-end.
///
/// The text prefill .mlpackage was traced with **fixed T=16** (RangeDim did
/// not stick through tracing — see docs/conversion-notes.md). We pad/truncate
/// inputs to 16 tokens here. This matches the smoke test in demo_smoke.py.
public final class LLMRunner: @unchecked Sendable {
    public struct Output {
        /// Per-step conditioning hidden states that the diffusion head consumes.
        /// Each entry is shape (1, T_chunk, 896).
        public var hiddenStates: [MLMultiArray]
        /// Prefill K/V cache (shape (L,1,KVH,T,D)). Nil on prefill-only paths
        /// that don't intend to decode further.
        public var kCache: MLMultiArray?
        public var vCache: MLMultiArray?
        /// Number of tokens represented in the cache (== T).
        public var cacheLength: Int
    }

    /// Baked-in prefill sequence length (trace-time).
    public static let prefillT = 16
    /// Pad token used to fill the prefill window. BOS=151643 keeps the LM in a
    /// well-conditioned regime without injecting semantic tokens.
    public static let padToken: Int32 = 151643

    private let textPrefill: MLModel
    private let textDecode: MLModel?
    private let ttsPrefill: MLModel?
    private let ttsDecode: MLModel?

    // NOTE: decode input arrays are allocated **per call** rather than kept as
    // stored properties. Earlier revisions pre-allocated (1,1) buffers on `self`
    // and mutated them in-place inside each step to save allocation cost, but
    // that made `ttsDecodeStep` unsafe to call concurrently for the pos/neg
    // CFG branches — the two calls would race on the same scratch memory.
    // The per-call allocation is negligible (4 tiny MLMultiArrays, ~microseconds)
    // compared to the 15-20ms CoreML forward, and it unlocks ~15% total
    // speedup by letting TTSLoop do `async let posStep / async let negStep`.

    public init(modelsDir: URL, config: MLModelConfiguration) async throws {
        func load(_ name: String, required: Bool) async throws -> MLModel? {
            let url = modelsDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                if required {
                    throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
                }
                return nil
            }
            return try await ModelLoader.load(packageURL: url, configuration: config)
        }
        self.textPrefill = try await load("text_lm_prefill.mlpackage", required: true)!
        self.textDecode  = try await load("text_lm_decode.mlpackage",  required: false)
        self.ttsPrefill  = try await load("tts_lm_prefill.mlpackage",  required: false)
        self.ttsDecode   = try await load("tts_lm_decode.mlpackage",   required: false)
    }

    // MARK: - Per-call scratch allocation (concurrency-safe)

    private static func makeIDs(_ value: Int32) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, 1], dataType: .int32)
        a.dataPointer.assumingMemoryBound(to: Int32.self)[0] = value
        return a
    }
    private static func makeTail(_ value: Float) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        a.dataPointer.assumingMemoryBound(to: Float.self)[0] = value
        return a
    }

    /// One *text* LM decode step (plain Qwen2 graph).
    public func textDecodeStep(
        inputID: Int32,
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        guard let model = textDecode else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "text decode model not loaded"])
        }
        let ids = try Self.makeIDs(inputID)
        let pos = try Self.makeIDs(positionID)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids, "k_cache": kCache, "v_cache": vCache, "position_ids": pos,
        ])
        let out = try await model.prediction(from: provider)
        return try Self.unpackDecode(out)
    }

    /// One *TTS* LM decode step. Matches the mask-form graph from
    /// `02b_convert_tts_lm.py` (TTSDecodeWrapper):
    ///
    ///   inputs_embeds = tok_embed * (1 - tail_mask) + lm_hidden * tail_mask
    ///                 + tts_input_types(tts_text_masks)
    ///
    /// - Parameters:
    ///   - inputID: token id. At speech-frame steps upstream passes 1 as a
    ///       sentinel; the embedding gets overwritten by `lmHidden` anyway
    ///       when `tailMask == 1`.
    ///   - lmHidden: (1, 1, 896) — text LM hidden (at text step) OR
    ///       `acoustic_connector(latent)` output (at speech step).
    ///   - tailMask: 1.0 if the slot should be overridden by `lmHidden`,
    ///       0.0 if the plain token embedding should pass through. Upstream
    ///       only overrides the tail, but at decode the step *is* the tail.
    ///   - textMask: 1 = text position, 0 = speech position. Drives
    ///       `tts_input_types` embedding.
    public func ttsDecodeStep(
        inputID: Int32,
        lmHidden: MLMultiArray,
        tailMask: Float,
        textMask: Int32,
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        guard let model = ttsDecode else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "tts decode model not loaded"])
        }
        // Per-call scratch — required so two concurrent calls (pos/neg CFG
        // branches) don't clobber each other's inputs mid-flight.
        let ids = try Self.makeIDs(inputID)
        let tail = try Self.makeTail(tailMask)
        let tm = try Self.makeIDs(textMask)
        let pos = try Self.makeIDs(positionID)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids,
            "lm_hidden_padded": lmHidden,
            "tail_mask": tail,
            "tts_text_masks": tm,
            "k_cache": kCache,
            "v_cache": vCache,
            "position_ids": pos,
        ])
        let out = try await model.prediction(from: provider)
        return try Self.unpackDecode(out)
    }

    private static func unpackDecode(_ out: MLFeatureProvider) throws
        -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray)
    {
        guard
            let h = out.featureValue(for: "hidden_states")?.multiArrayValue,
            let kNew = out.featureValue(for: "k_cache_out")?.multiArrayValue,
            let vNew = out.featureValue(for: "v_cache_out")?.multiArrayValue
        else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "decode outputs missing"])
        }
        return (h, kNew, vNew)
    }

    /// TTS LM prefill (mask form). Pads tokens to T=16 and passes the
    /// caller's `lmHiddenPadded` (same length) as the tail-splice hidden, with
    /// `tailMask` = all 1s (full override) and `ttsTextMasks` = all 1s (text).
    ///
    /// - Parameter zeroLM: if true, the LM-hidden input is zeroed instead of
    ///     `lmHiddenPadded`. Used to build the negative (uncond) TTS KV cache
    ///     for CFG.
    public func runTTSPrefill(
        tokens: [Int32],
        lmHiddenPadded: MLMultiArray,
        zeroLM: Bool
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        guard let model = ttsPrefill else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "tts prefill model not loaded"])
        }
        let T = Self.prefillT
        var padded = Array(tokens.prefix(T))
        while padded.count < T { padded.append(Self.padToken) }

        let ids = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let idPtr = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<T { idPtr[i] = padded[i] }

        let lm: MLMultiArray
        if zeroLM {
            lm = try MLMultiArray(shape: [1, NSNumber(value: T), 896], dataType: .float32)
            memset(lm.dataPointer, 0, T * 896 * MemoryLayout<Float>.size)
        } else {
            lm = lmHiddenPadded
        }
        let tail = try MLMultiArray(shape: [1, NSNumber(value: T), 1], dataType: .float32)
        let tailPtr = tail.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<T { tailPtr[i] = 1.0 }
        let tm = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let tmPtr = tm.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<T { tmPtr[i] = 1 }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids,
            "lm_hidden_padded": lm,
            "tail_mask": tail,
            "tts_text_masks": tm,
        ])
        let out = try await model.prediction(from: provider)
        guard
            let h = out.featureValue(for: "hidden_states")?.multiArrayValue,
            let k = out.featureValue(for: "k_cache")?.multiArrayValue,
            let v = out.featureValue(for: "v_cache")?.multiArrayValue
        else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "tts prefill missing outputs"])
        }
        return (h, k, v)
    }

    /// Prefill-only path: pad/truncate to T=16, run text LM prefill, and return
    /// the full hidden-state block as single-chunk conditioning.
    ///
    /// NOTE: this is *not* the real synthesis path. Real synthesis needs the
    /// TTS LM autoregressive decode loop to produce per-acoustic-step hidden
    /// states, gated by `tts_eos_classifier`. That's the next piece of work
    /// (see docs/conversion-notes.md "Phase E — Swift wiring"). For now this
    /// gets the graph-level end-to-end pipeline executing.
    public func runPrefillAndDecode(tokens: [Int32]) async throws -> Output {
        let T = Self.prefillT
        var padded = Array(tokens.prefix(T))
        while padded.count < T { padded.append(Self.padToken) }

        let ids = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let idPtr = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<T { idPtr[i] = padded[i] }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids])
        let out = try await textPrefill.prediction(from: provider)
        guard let hidden = out.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "missing hidden_states"])
        }
        let k = out.featureValue(for: "k_cache")?.multiArrayValue
        let v = out.featureValue(for: "v_cache")?.multiArrayValue
        return Output(hiddenStates: [hidden], kCache: k, vCache: v, cacheLength: T)
    }
}

enum NotImplementedError: Error {
    case phaseBPending
    case phaseCPending
    case phaseDPending
}
