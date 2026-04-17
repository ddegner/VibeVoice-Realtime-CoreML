import Foundation
import ArgumentParser
import VibeVoiceKit

@main
struct VibeVoiceCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vibevoice-cli",
        abstract: "Synthesize speech with the CoreML VibeVoice-Realtime pipeline."
    )

    @Argument(help: "Text to speak.")
    var text: String

    @Option(name: .shortAndLong, help: "Directory containing the .mlpackage artifacts + voices/.")
    var models: String

    @Option(name: .shortAndLong, help: "Output WAV path (24 kHz mono float).")
    var output: String = "./out.wav"

    @Option(help: "Voice name (e.g. en-Frank_man). Use --list-voices to see available.")
    var voice: String = "en-Frank_man"

    @Option(help: "Number of diffusion steps (default 5, upstream demo match).")
    var steps: Int = 5

    @Option(help: "CFG guidance scale (default 1.5, upstream demo match).")
    var guidance: Float = 1.5

    @Option(help: "Seed for diffusion noise (omit for non-deterministic).")
    var seed: UInt64?

    @Flag(help: "List available voices and exit.")
    var listVoices: Bool = false

    @Flag(help: "Also play the audio after synthesis.")
    var play: Bool = false

    @Flag(help: "Print per-frame EOS classifier probabilities (diagnostic).")
    var traceEOS: Bool = false

    func run() async throws {
        let modelsDir = URL(fileURLWithPath: models, isDirectory: true)
        var cfg = VibeVoiceEngine.Config(modelsDir: modelsDir)
        cfg.diffusionSteps = steps
        cfg.guidanceScale = guidance

        print("Loading engine from \(modelsDir.path)…")
        let engine = try await VibeVoiceEngine(config: cfg)
        if let seed = seed {
            engine.setSeed(seed)
            print("Diffusion seed: \(seed)")
        }

        if listVoices {
            let voices = engine.availableVoices()
            print("Available voices: \(voices.joined(separator: ", "))")
            return
        }

        print("Loading voice: \(voice)")
        let voicePrompt = try engine.loadVoice(voice)
        print("  lm cache: \(voicePrompt.lm.length) tokens, tts_lm cache: \(voicePrompt.ttsLM.length) tokens")

        print("Synthesizing: \(text.prefix(80))\(text.count > 80 ? "…" : "")")
        var loopConfig = TTSLoop.Config()
        loopConfig.logEOSTrajectory = traceEOS
        let (samples, timing) = try await engine.synthesize(
            text: text, voicePrompt: voicePrompt, loopConfig: loopConfig)
        let audioDur = Double(samples.count) / 24_000.0
        print("Generated \(samples.count) samples (\(String(format: "%.2f", audioDur)) s)")
        print(timing)
        let rtf = timing.total / audioDur
        print(String(format: "Real-time factor: %.2fx (%.2fs wall / %.2fs audio)", rtf, timing.total, audioDur))

        let outURL = URL(fileURLWithPath: output)
        try AudioPlayer.writeWAV(samples, to: outURL)
        print("Wrote \(outURL.path)")

        if play {
            let p = try AudioPlayer()
            p.play(samples)
            try? await Task.sleep(nanoseconds: UInt64(Double(samples.count) / 24_000.0 * 1e9) + 500_000_000)
        }
    }
}
