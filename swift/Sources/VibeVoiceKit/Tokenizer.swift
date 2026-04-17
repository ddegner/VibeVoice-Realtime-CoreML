import Foundation
import Tokenizers

/// Wraps the Qwen2.5 tokenizer (loaded via swift-transformers) for the text
/// LM. VibeVoice-Realtime reuses Qwen2.5's tokenizer verbatim.
public final class VibeVoiceTokenizer {
    private let impl: Tokenizer

    public init(modelsDir: URL) async throws {
        // Expect tokenizer.json + tokenizer_config.json bundled alongside the
        // .mlpackages. The conversion scripts save these next to model_manifest.json.
        let tokenizerDir = modelsDir.appendingPathComponent("tokenizer")
        self.impl = try await AutoTokenizer.from(modelFolder: tokenizerDir)
    }

    /// Tokenize text matching upstream's streaming processor behavior:
    ///   tokenizer.encode(text.strip() + "\n", add_special_tokens=False)
    /// The trailing "\n" is significant. On BPE tokenizers (Qwen2.5 included),
    /// a trailing "." followed by "\n" merges into a single token (e.g. 624
    /// for ".\n") — *different* from the token for "." alone (13). Getting
    /// this right matters because the last text token of each 5-token window
    /// becomes the TTS-LM conditioning for the subsequent speech-diffusion
    /// pass; using the wrong token drifts the acoustic feedback loop and
    /// produces word-repetition / stuttering in later windows.
    public func encode(_ text: String) throws -> [Int32] {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        return impl.encode(text: normalized).map(Int32.init)
    }

    public func decode(_ tokens: [Int32]) -> String {
        impl.decode(tokens: tokens.map(Int.init))
    }
}
