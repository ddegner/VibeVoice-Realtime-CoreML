import Foundation
import CoreML

/// Top-level orchestration for the VibeVoice CoreML pipeline.
///
/// The flow:
///   1. Tokenize text.
///   2. Load voice prompt (pre-computed KV caches for voice identity).
///   3. Interleaved text-window / speech-window TTS loop with CFG.
///   4. Diffusion per frame → acoustic latents.
///   5. VAE decoder → 24 kHz PCM.
public final class VibeVoiceEngine: @unchecked Sendable {
    /// Identifies a model so per-kind compute-unit overrides can target it.
    public enum ModelKind: Sendable, Hashable {
        case textLM        // text_lm_decode (LM, KV cache, dynamic shape)
        case ttsLM         // tts_lm_decode (LM, KV cache, dynamic shape)
        case diffusion     // diffusion_head (small fp16, fixed shape)
        case vae           // vae_decoder (fixed shape T=16)
        case connector     // acoustic_connector (tiny fp16, fixed shape)
        case eos           // tts_eos_classifier (tiny fp16, fixed shape)
    }

    public struct Config: Sendable {
        public var modelsDir: URL
        public var diffusionSteps: Int = 5
        public var guidanceScale: Float = 1.5
        /// Default compute units for any model not present in `computeUnitsByKind`.
        /// LMs need this — KV-cache shape changes per call so ANE can't compile
        /// a single plan, and falls back to GPU anyway with extra dispatch cost.
        public var computeUnits: MLComputeUnits = .cpuAndGPU
        /// Per-kind compute-unit overrides. Empty by default; populate to route
        /// fixed-shape heads to `.cpuAndNeuralEngine` (or `.all`) so the GPU
        /// stays free for the LMs on the critical path. Phase 3 sweep matrix.
        public var computeUnitsByKind: [ModelKind: MLComputeUnits] = [:]
        /// Allow fp16 accumulation on the GPU for the LLM / VAE / heads.
        /// These models are robust to the extra rounding; matters only for
        /// ~5-10% throughput.
        public var lowPrecisionAccumForLLMs: Bool = true
        /// Allow fp16 accumulation on the *diffusion head*. Much more
        /// sensitive — small extra error here feeds back into the TTS-LM via
        /// the acoustic connector and can cause word-repetition / stuttering
        /// at the 5-step count. Default `false` for quality.
        public var lowPrecisionAccumForDiffusion: Bool = false
        /// Run a single zero-input prediction through the heads + VAE in
        /// parallel right after model load, to absorb first-call CoreML
        /// compile/plan latency off the synthesis critical path. The two
        /// LM-decode models and the diffusion head are skipped because
        /// their input shapes depend on the voice prompt / per-frame state.
        public var warmupAtInit: Bool = true
        /// Number of `tts_lm_decode` MLModel handles. The pos/neg CFG
        /// branches alternate handles. Default 1.
        ///
        /// Phase 5 measurement (2025-04, M-series): setting this to 2 did
        /// NOT improve wall time. The two branches already share the GPU,
        /// and a second handle adds load (+6.85s on the 168s reference clip)
        /// rather than unlocking concurrency. Audio remained bit-identical.
        /// Kept as a knob in case future hardware changes the trade.
        public var ttsHandles: Int = 1
        /// Benchmark switch: load independent CoreML artifacts concurrently.
        /// Default stays sequential until load-time and peak-memory behavior is
        /// measured across supported machines.
        public var parallelModelLoading: Bool = false
        public init(modelsDir: URL) { self.modelsDir = modelsDir }
    }

    private let config: Config
    let llm: LLMRunner
    let diffusion: DiffusionRunner
    let vae: VAERunner
    let heads: HeadsRunner
    private let tokenizer: VibeVoiceTokenizer

    public init(config: Config) async throws {
        self.config = config

        // Per-kind compute units: lookup honors `computeUnitsByKind` first,
        // falls back to `config.computeUnits`. Builds one MLModelConfiguration
        // per kind so heads can route to ANE while LMs stay on GPU.
        func cfg(_ kind: ModelKind, lowPrecision: Bool) -> MLModelConfiguration {
            let mc = MLModelConfiguration()
            mc.computeUnits = config.computeUnitsByKind[kind] ?? config.computeUnits
            mc.allowLowPrecisionAccumulationOnGPU = lowPrecision
            return mc
        }

        let textCfg      = cfg(.textLM,    lowPrecision: config.lowPrecisionAccumForLLMs)
        let ttsCfg       = cfg(.ttsLM,     lowPrecision: config.lowPrecisionAccumForLLMs)
        let diffCfg      = cfg(.diffusion, lowPrecision: config.lowPrecisionAccumForDiffusion)
        let vaeCfg       = cfg(.vae,       lowPrecision: config.lowPrecisionAccumForLLMs)
        let connectorCfg = cfg(.connector, lowPrecision: config.lowPrecisionAccumForLLMs)
        let eosCfg       = cfg(.eos,       lowPrecision: config.lowPrecisionAccumForLLMs)

        if config.parallelModelLoading {
            async let tokenizerTask = VibeVoiceTokenizer(modelsDir: config.modelsDir)
            async let llmTask = LLMRunner(
                modelsDir: config.modelsDir, textConfig: textCfg, ttsConfig: ttsCfg,
                ttsHandles: config.ttsHandles)
            async let diffusionTask = DiffusionRunner(
                modelURL: config.modelsDir.appendingPathComponent("diffusion_head.mlpackage"),
                mlConfig: diffCfg,
                steps: config.diffusionSteps,
                guidanceScale: config.guidanceScale)
            async let vaeTask = VAERunner(
                modelURL: config.modelsDir.appendingPathComponent("vae_decoder.mlpackage"),
                mlConfig: vaeCfg)
            async let headsTask = HeadsRunner(
                modelsDir: config.modelsDir, connectorConfig: connectorCfg, eosConfig: eosCfg)
            let loaded = try await (tokenizerTask, llmTask, diffusionTask, vaeTask, headsTask)
            self.tokenizer = loaded.0
            self.llm = loaded.1
            self.diffusion = loaded.2
            self.vae = loaded.3
            self.heads = loaded.4
        } else {
            self.tokenizer = try await VibeVoiceTokenizer(modelsDir: config.modelsDir)
            self.llm = try await LLMRunner(
                modelsDir: config.modelsDir, textConfig: textCfg, ttsConfig: ttsCfg,
                ttsHandles: config.ttsHandles)
            self.diffusion = try await DiffusionRunner(
                modelURL: config.modelsDir.appendingPathComponent("diffusion_head.mlpackage"),
                mlConfig: diffCfg,
                steps: config.diffusionSteps,
                guidanceScale: config.guidanceScale)
            self.vae = try await VAERunner(
                modelURL: config.modelsDir.appendingPathComponent("vae_decoder.mlpackage"),
                mlConfig: vaeCfg)
            self.heads = try await HeadsRunner(
                modelsDir: config.modelsDir, connectorConfig: connectorCfg, eosConfig: eosCfg)
        }

        if config.warmupAtInit {
            // Capture local refs so the async-let tasks don't have to send
            // `self` (Swift 6 strict-concurrency rejects that — `self` is
            // task-isolated until init returns). Heads + VAE are class
            // references, safe to share.
            let heads = self.heads
            let vae = self.vae
            async let h: Void = heads.warmup()
            async let v: Void = vae.warmup()
            _ = try await (h, v)
        }
    }

    /// Expose the tokenizer so callers (CLI, tests, parity harnesses) can
    /// inspect the exact token stream fed into the generator — useful for
    /// verifying parity against the Python reference dump.
    public func tokenize(_ text: String) throws -> [Int32] {
        try tokenizer.encode(text)
    }

    /// Synthesize with a voice prompt. Returns 24 kHz Float32 PCM.
    public func synthesize(
        text: String,
        voicePrompt: VoicePrompt,
        loopConfig: TTSLoop.Config = TTSLoop.Config()
    ) async throws -> (samples: [Float], timing: TTSLoop.Timing) {
        try await synthesize(text: text, voicePrompt: voicePrompt, loopConfig: loopConfig, seed: nil)
    }

    /// Synthesize with a diffusion seed that applies to this call only.
    /// Unlike `setSeed`, concurrent calls with different seeds cannot race on
    /// the shared default. `nil` uses the engine-wide seed from `setSeed`.
    public func synthesize(
        text: String,
        voicePrompt: VoicePrompt,
        loopConfig: TTSLoop.Config = TTSLoop.Config(),
        seed: UInt64?
    ) async throws -> (samples: [Float], timing: TTSLoop.Timing) {
        let tokens = try tokenizer.encode(text)
        let loop = TTSLoop(llm: llm, diffusion: diffusion, vae: vae, heads: heads)
        let samples = try await loop.generate(
            tokens: tokens, voicePrompt: voicePrompt, config: loopConfig, seed: seed)
        return (samples, loop.lastTiming)
    }

    /// Seed the diffusion noise RNG. Same seed + same text + same voice →
    /// bit-identical output. Pass `nil` to revert to non-deterministic noise.
    public func setSeed(_ seed: UInt64?) {
        diffusion.setSeed(seed)
    }

    /// List available voices from the voices directory.
    public func availableVoices() -> [String] {
        let fm = FileManager.default
        let voicesDir = config.modelsDir.appendingPathComponent("voices").path
        guard let names = try? fm.contentsOfDirectory(atPath: voicesDir) else { return [] }
        // Path-based enumeration + per-entry `fileExists(isDirectory:)` is
        // more robust than URL resource-value caching inside a signed .app
        // bundle, where `.isDirectoryKey` has been observed to come back nil.
        return names.filter { name in
            var isDir: ObjCBool = false
            let full = (voicesDir as NSString).appendingPathComponent(name)
            return fm.fileExists(atPath: full, isDirectory: &isDir) && isDir.boolValue
        }.sorted()
    }

    /// Load a voice prompt by name (e.g. "en-Frank_man").
    public func loadVoice(_ name: String) throws -> VoicePrompt {
        let dir = config.modelsDir
            .appendingPathComponent("voices")
            .appendingPathComponent(name)
        return try VoicePrompt(voiceDir: dir)
    }
}
