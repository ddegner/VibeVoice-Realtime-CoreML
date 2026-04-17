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
public final class VibeVoiceEngine {
    public struct Config: Sendable {
        public var modelsDir: URL
        public var diffusionSteps: Int = 5
        public var guidanceScale: Float = 1.5
        public var computeUnits: MLComputeUnits = .cpuAndGPU
        /// Allow fp16 accumulation on the GPU for the LLM / VAE / heads.
        /// These models are robust to the extra rounding; matters only for
        /// ~5-10% throughput.
        public var lowPrecisionAccumForLLMs: Bool = true
        /// Allow fp16 accumulation on the *diffusion head*. Much more
        /// sensitive — small extra error here feeds back into the TTS-LM via
        /// the acoustic connector and can cause word-repetition / stuttering
        /// at the 5-step count. Default `false` for quality.
        public var lowPrecisionAccumForDiffusion: Bool = false
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

        // Shared fp16-accum config used for the robust models.
        let fp16Config = MLModelConfiguration()
        fp16Config.computeUnits = config.computeUnits
        fp16Config.allowLowPrecisionAccumulationOnGPU = config.lowPrecisionAccumForLLMs

        // Dedicated config for the diffusion head (quality-sensitive).
        let diffConfig = MLModelConfiguration()
        diffConfig.computeUnits = config.computeUnits
        diffConfig.allowLowPrecisionAccumulationOnGPU = config.lowPrecisionAccumForDiffusion

        self.tokenizer = try await VibeVoiceTokenizer(modelsDir: config.modelsDir)
        self.llm = try await LLMRunner(modelsDir: config.modelsDir, config: fp16Config)
        self.diffusion = try await DiffusionRunner(
            modelURL: config.modelsDir.appendingPathComponent("diffusion_head.mlpackage"),
            mlConfig: diffConfig,
            steps: config.diffusionSteps,
            guidanceScale: config.guidanceScale
        )
        self.vae = try await VAERunner(
            modelURL: config.modelsDir.appendingPathComponent("vae_decoder.mlpackage"),
            mlConfig: fp16Config
        )
        self.heads = try await HeadsRunner(modelsDir: config.modelsDir, config: fp16Config)
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
        let tokens = try tokenizer.encode(text)
        let loop = TTSLoop(llm: llm, diffusion: diffusion, vae: vae, heads: heads)
        let samples = try await loop.generate(
            tokens: tokens, voicePrompt: voicePrompt, config: loopConfig)
        return (samples, loop.lastTiming)
    }

    /// Seed the diffusion noise RNG. Same seed + same text + same voice →
    /// bit-identical output. Pass `nil` to revert to non-deterministic noise.
    public func setSeed(_ seed: UInt64?) {
        diffusion.setSeed(seed)
    }

    /// List available voices from the voices directory.
    public func availableVoices() -> [String] {
        let voicesDir = config.modelsDir.appendingPathComponent("voices")
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: voicesDir, includingPropertiesForKeys: nil
        ) else { return [] }
        return contents
            .filter { $0.hasDirectoryPath }
            .map(\.lastPathComponent)
            .sorted()
    }

    /// Load a voice prompt by name (e.g. "en-Frank_man").
    public func loadVoice(_ name: String) throws -> VoicePrompt {
        let dir = config.modelsDir
            .appendingPathComponent("voices")
            .appendingPathComponent(name)
        return try VoicePrompt(voiceDir: dir)
    }
}
