import Foundation
import CoreML

/// Wraps the two small heads exported by `conversion/scripts/06_convert_heads.py`:
///
///   acoustic_connector    (1, T, 64)  → (1, T, 896)
///   tts_eos_classifier    (1, 896)    → (1, 1)   (sigmoid prob)
///
/// Plus the two scalar normalization constants persisted to `speech_constants.json`.
/// The streaming TTS loop needs all three: per-frame speech latent is projected
/// into LM hidden space (connector), used to update the TTS LM's rolling
/// conditioning, and EOS is checked off the TTS LM's last hidden.
public final class HeadsRunner: @unchecked Sendable {
    public struct SpeechConstants: Sendable, Codable {
        public let speech_scaling_factor: Float
        public let speech_bias_factor: Float
    }

    public let connector: MLModel
    public let eos: MLModel
    public let constants: SpeechConstants

    public init(modelsDir: URL, config: MLModelConfiguration) async throws {
        self.connector = try await ModelLoader.load(
            packageURL: modelsDir.appendingPathComponent("acoustic_connector.mlpackage"),
            configuration: config)
        self.eos = try await ModelLoader.load(
            packageURL: modelsDir.appendingPathComponent("tts_eos_classifier.mlpackage"),
            configuration: config)

        let constURL = modelsDir.appendingPathComponent("speech_constants.json")
        let data = try Data(contentsOf: constURL)
        // speech_constants.json has a "note" field too — decode loosely.
        struct Raw: Codable {
            let speech_scaling_factor: Float
            let speech_bias_factor: Float
        }
        let raw = try JSONDecoder().decode(Raw.self, from: data)
        // scale is used as a divisor in normalizeLatentForVAE and in
        // TTSLoop.decodeWindow — a zero or non-finite value would cascade NaN
        // through the VAE for every frame.
        guard raw.speech_scaling_factor.isFinite,
              raw.speech_scaling_factor != 0,
              raw.speech_bias_factor.isFinite else {
            throw NSError(domain: "VibeVoiceKit.HeadsRunner", code: 3, userInfo: [
                NSLocalizedDescriptionKey:
                    "speech_constants.json has invalid values: scale=\(raw.speech_scaling_factor), bias=\(raw.speech_bias_factor)"
            ])
        }
        self.constants = SpeechConstants(
            speech_scaling_factor: raw.speech_scaling_factor,
            speech_bias_factor: raw.speech_bias_factor)
    }

    /// Project a speech latent into LM hidden space.
    /// - Parameter latent: (1, T, 64)
    /// - Returns: (1, T, 896)
    public func project(latent: MLMultiArray) async throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["latent": latent])
        let out = try await connector.prediction(from: provider)
        guard let v = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw NSError(domain: "VibeVoiceKit.HeadsRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "connector missing output"])
        }
        return v
    }

    /// Run the EOS head on a (1, 896) hidden state; returns sigmoid prob (0…1).
    public func eosProbability(hidden: MLMultiArray) async throws -> Float {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["hidden": hidden])
        let out = try await eos.prediction(from: provider)
        guard let p = out.featureValue(for: "prob")?.multiArrayValue else {
            throw NSError(domain: "VibeVoiceKit.HeadsRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "eos missing output"])
        }
        return p.dataPointer.assumingMemoryBound(to: Float.self)[0]
    }

    /// Apply `scaled = latent / scale - bias` element-wise in place of the raw
    /// VAE input. Matches `modeling_vibevoice_streaming_inference.py:783`.
    public func normalizeLatentForVAE(_ latent: MLMultiArray) throws -> MLMultiArray {
        let scale = constants.speech_scaling_factor
        let bias = constants.speech_bias_factor
        let out = try MLMultiArray(shape: latent.shape, dataType: .float32)
        let src = latent.dataPointer.assumingMemoryBound(to: Float.self)
        let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<latent.count { dst[i] = src[i] / scale - bias }
        return out
    }
}
