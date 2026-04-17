// SwiftUI app target. Add this source file to an Xcode project with a macOS
// target (15+) and a dependency on the VibeVoiceKit SPM package — it's not
// wired into Package.swift because SwiftPM doesn't build app bundles.
//
// Phase F. Intentionally minimal.

import SwiftUI
import VibeVoiceKit

@main
struct VibeVoiceApp: App {
    var body: some Scene {
        WindowGroup("VibeVoice") {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var text: String = "Hello from CoreML."
    @State private var modelsDir: String = ""
    @State private var status: String = "idle"
    @State private var engine: VibeVoiceEngine?
    @State private var player: AudioPlayer?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Models directory", text: $modelsDir)
            TextEditor(text: $text)
                .frame(minHeight: 120)
                .border(.secondary)
            HStack {
                Button("Generate & Play") { Task { await generate() } }
                    .disabled(modelsDir.isEmpty)
                Button("Stop") { player?.stop() }
            }
            Text(status).font(.caption).foregroundStyle(.secondary)
        }
        .padding()
        .frame(minWidth: 480, minHeight: 360)
    }

    private func generate() async {
        do {
            status = "loading models…"
            let cfg = VibeVoiceEngine.Config(modelsDir: URL(fileURLWithPath: modelsDir))
            let engine = try await VibeVoiceEngine(config: cfg)
            self.engine = engine
            status = "synthesizing…"
            let samples = try await engine.synthesize(text: text)
            status = "playing \(String(format: "%.1f", Double(samples.count)/24_000.0))s"
            let p = try AudioPlayer()
            self.player = p
            p.play(samples)
        } catch {
            status = "error: \(error)"
        }
    }
}
