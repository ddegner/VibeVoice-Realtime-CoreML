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

    @Option(help: "Hard cap on speech frames generated (default 200 ≈ 26.6s; 4500 ≈ 10 min).")
    var maxFrames: Int = 200

    @Flag(help: "Emit a single JSON line with per-stage timings + audio stats + run metadata. Suppresses the human progress prints.")
    var jsonTiming: Bool = false

    @Option(help: "Free-form label embedded in --json-timing output (e.g. phase identifier).")
    var label: String?

    @Option(help: "Per-kind compute-unit overrides for the Phase 3 ANE sweep. Comma-separated kind=unit pairs. Kinds: textLM, ttsLM, diffusion, vae, connector, eos. Units: cpu, gpu, ane, all (alias of cpuANE), cpuGPU. Example: --compute-units 'vae=ane,connector=ane,eos=ane'.")
    var computeUnits: String?

    @Option(help: "Phase 5 CFG multiplex: number of tts_lm_decode MLModel handles to load (default 1). Set 2 to alternate pos/neg branches.")
    var ttsHandles: Int = 1

    func run() async throws {
        let quiet = jsonTiming
        func say(_ s: @autoclosure () -> String) { if !quiet { print(s()) } }

        let modelsDir = URL(fileURLWithPath: models, isDirectory: true)
        var cfg = VibeVoiceEngine.Config(modelsDir: modelsDir)
        cfg.diffusionSteps = steps
        cfg.guidanceScale = guidance
        if let raw = computeUnits {
            cfg.computeUnitsByKind = try parseComputeUnits(raw)
        }
        cfg.ttsHandles = ttsHandles

        say("Loading engine from \(modelsDir.path)…")
        let engine = try await VibeVoiceEngine(config: cfg)
        if let seed = seed {
            engine.setSeed(seed)
            say("Diffusion seed: \(seed)")
        }

        if listVoices {
            let voices = engine.availableVoices()
            print("Available voices: \(voices.joined(separator: ", "))")
            return
        }

        say("Loading voice: \(voice)")
        let voicePrompt = try engine.loadVoice(voice)
        say("  lm cache: \(voicePrompt.lm.length) tokens, tts_lm cache: \(voicePrompt.ttsLM.length) tokens")

        say("Synthesizing: \(text.prefix(80))\(text.count > 80 ? "…" : "")")
        var loopConfig = TTSLoop.Config()
        loopConfig.logEOSTrajectory = traceEOS
        loopConfig.maxFrames = maxFrames
        let (samples, timing) = try await engine.synthesize(
            text: text, voicePrompt: voicePrompt, loopConfig: loopConfig)
        let audioDur = Double(samples.count) / 24_000.0
        say("Generated \(samples.count) samples (\(String(format: "%.2f", audioDur)) s)")
        say("\(timing)")
        let rtf = timing.total / audioDur
        say(String(format: "Real-time factor: %.2fx (%.2fs wall / %.2fs audio)", rtf, timing.total, audioDur))

        let outURL = URL(fileURLWithPath: output)
        try AudioPlayer.writeWAV(samples, to: outURL)
        say("Wrote \(outURL.path)")

        if jsonTiming {
            let stats = AudioStats.compute(samples)
            let line = JSONTimingLine.encode(
                label: label,
                voice: voice,
                seed: seed,
                charCount: text.count,
                steps: steps,
                guidance: guidance,
                maxFrames: maxFrames,
                samples: samples.count,
                timing: timing,
                stats: stats)
            print(line)
        }

        if play {
            let p = try AudioPlayer()
            p.play(samples)
            try? await Task.sleep(nanoseconds: UInt64(Double(samples.count) / 24_000.0 * 1e9) + 500_000_000)
        }
    }
}

// MARK: - Compute-unit parsing

import CoreML

func parseComputeUnits(_ raw: String) throws -> [VibeVoiceEngine.ModelKind: MLComputeUnits] {
    var out: [VibeVoiceEngine.ModelKind: MLComputeUnits] = [:]
    for pair in raw.split(separator: ",") {
        let kv = pair.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        guard kv.count == 2 else {
            throw ValidationError("--compute-units pair '\(pair)' must be kind=unit")
        }
        let kind: VibeVoiceEngine.ModelKind
        switch kv[0] {
        case "textLM":    kind = .textLM
        case "ttsLM":     kind = .ttsLM
        case "diffusion": kind = .diffusion
        case "vae":       kind = .vae
        case "connector": kind = .connector
        case "eos":       kind = .eos
        default: throw ValidationError("--compute-units unknown kind '\(kv[0])'")
        }
        let unit: MLComputeUnits
        switch kv[1].lowercased() {
        case "cpu":     unit = .cpuOnly
        case "gpu":     unit = .cpuAndGPU   // CoreML has no "GPU only"; cpuAndGPU is the canonical "prefer GPU" option
        case "cpugpu":  unit = .cpuAndGPU
        case "ane":     unit = .cpuAndNeuralEngine
        case "all":     unit = .all
        default: throw ValidationError("--compute-units unknown unit '\(kv[1])'")
        }
        out[kind] = unit
    }
    return out
}

struct ValidationError: Error, CustomStringConvertible {
    let message: String
    init(_ m: String) { message = m }
    var description: String { message }
}

// MARK: - Audio stats

/// Quality-gate metrics computed from final 24 kHz PCM. Matches the criteria
/// in `/Users/degner/.claude/plans/purrfect-pondering-karp.md` ("Quality gate
/// criteria") so `bench/compare` can pass/fail a run without re-deriving them.
struct AudioStats {
    let peak: Float
    let rms: Float
    let longestSilentRunSec: Double
    let q1RMS: Float
    let q4RMS: Float
    let q4OverQ1: Float
    let glitchWindows: Int
    let durationSec: Double

    static func compute(_ samples: [Float], sampleRate: Int = 24_000) -> AudioStats {
        guard !samples.isEmpty else {
            return .init(peak: 0, rms: 0, longestSilentRunSec: 0,
                         q1RMS: 0, q4RMS: 0, q4OverQ1: 0,
                         glitchWindows: 0, durationSec: 0)
        }
        var peak: Float = 0
        var sumSq: Double = 0
        for s in samples {
            let a = abs(s)
            if a > peak { peak = a }
            sumSq += Double(s) * Double(s)
        }
        let rms = Float((sumSq / Double(samples.count)).squareRoot())

        // Silent-run detection: contiguous samples below 0.005 amplitude.
        let silenceThreshold: Float = 0.005
        var longestRun = 0
        var run = 0
        for s in samples {
            if abs(s) < silenceThreshold {
                run += 1
                if run > longestRun { longestRun = run }
            } else {
                run = 0
            }
        }
        let longestSilentRunSec = Double(longestRun) / Double(sampleRate)

        // Q1 vs Q4 RMS — first 25% vs last 25% of the clip.
        let q = samples.count / 4
        let q1 = rmsOf(samples, range: 0..<q)
        let q4 = rmsOf(samples, range: (samples.count - q)..<samples.count)
        let q4OverQ1 = q1 > 0 ? q4 / q1 : 0

        // Glitch windows: 10ms windows with peak > 5× the median window peak
        // AND > 0.2 amplitude. Cheap stand-in for click/pop detector.
        let win = max(1, sampleRate / 100)
        var winPeaks: [Float] = []
        winPeaks.reserveCapacity(samples.count / win + 1)
        var i = 0
        while i < samples.count {
            let end = min(i + win, samples.count)
            var p: Float = 0
            for j in i..<end { let a = abs(samples[j]); if a > p { p = a } }
            winPeaks.append(p)
            i = end
        }
        let sorted = winPeaks.sorted()
        let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
        let glitchThresh = max(median * 5, 0.2)
        let glitches = winPeaks.reduce(0) { $0 + ($1 > glitchThresh ? 1 : 0) }

        return .init(
            peak: peak, rms: rms,
            longestSilentRunSec: longestSilentRunSec,
            q1RMS: q1, q4RMS: q4, q4OverQ1: q4OverQ1,
            glitchWindows: glitches,
            durationSec: Double(samples.count) / Double(sampleRate))
    }

    private static func rmsOf(_ s: [Float], range: Range<Int>) -> Float {
        guard !range.isEmpty else { return 0 }
        var sumSq: Double = 0
        for i in range { sumSq += Double(s[i]) * Double(s[i]) }
        return Float((sumSq / Double(range.count)).squareRoot())
    }
}

// MARK: - JSON line encoder

/// One-line JSON output for `--json-timing`. Stable schema — `bench/compare`
/// reads it back; if you add fields, add at the end.
enum JSONTimingLine {
    static func encode(
        label: String?,
        voice: String,
        seed: UInt64?,
        charCount: Int,
        steps: Int,
        guidance: Float,
        maxFrames: Int,
        samples: Int,
        timing: TTSLoop.Timing,
        stats: AudioStats
    ) -> String {
        let iso = ISO8601DateFormatter().string(from: Date())
        let dict: [String: Any] = [
            "schema": 1,
            "ts": iso,
            "git_sha": gitSHA() ?? NSNull(),
            "label": label ?? NSNull(),
            "voice": voice,
            "seed": seed.map { Int64(bitPattern: $0) } as Any? ?? NSNull(),
            "char_count": charCount,
            "diffusion_steps": steps,
            "guidance": Double(guidance),
            "max_frames": maxFrames,
            "samples": samples,
            "audio_duration_sec": stats.durationSec,
            "timing": [
                "total":           timing.total,
                "text_lm_decode":  timing.textLMDecode,
                "tts_lm_decode":   timing.ttsLMDecode,
                "diffusion":       timing.diffusion,
                "connector":       timing.connector,
                "eos_classifier":  timing.eos,
                "vae_decode":      timing.vaeDecode,
                "speech_frames":   timing.speechFrames,
                "diffusion_inner_steps": timing.diffusionSteps,
            ] as [String: Any],
            "audio_stats": [
                "peak":                  Double(stats.peak),
                "rms":                   Double(stats.rms),
                "longest_silent_run_s":  stats.longestSilentRunSec,
                "q1_rms":                Double(stats.q1RMS),
                "q4_rms":                Double(stats.q4RMS),
                "q4_over_q1":            Double(stats.q4OverQ1),
                "glitch_windows":        stats.glitchWindows,
            ] as [String: Any],
            "rt_factor": stats.durationSec > 0 ? timing.total / stats.durationSec : 0,
        ]
        let data = try! JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func gitSHA() -> String? {
        // Cheapest path: env var injected by the bench harness.
        if let s = ProcessInfo.processInfo.environment["GIT_SHA"], !s.isEmpty {
            return s
        }
        // Fallback: shell out (only when run interactively; fine for CLI).
        let p = Process()
        p.launchPath = "/usr/bin/env"
        p.arguments = ["git", "rev-parse", "--short", "HEAD"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let s = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (s?.isEmpty == false) ? s : nil
        } catch {
            return nil
        }
    }
}
