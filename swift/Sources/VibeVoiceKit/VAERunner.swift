import Foundation
import CoreML

/// Single-forward-pass VAE decoder: acoustic latents → 24 kHz PCM.
public final class VAERunner: @unchecked Sendable {
    private let model: MLModel

    public init(modelURL: URL, mlConfig: MLModelConfiguration) async throws {
        self.model = try await ModelLoader.load(packageURL: modelURL, configuration: mlConfig)
    }

    /// Takes latents shaped (1, 64, 16) — the VAE is traced at fixed T=16
    /// (reflection padding prevented dynamic T). 16 latent frames ≈ 2.133 s
    /// of 24 kHz audio (3200× upsampling confirmed at conversion).
    public func decode(latents: MLMultiArray) async throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["latent": latents])
        let out = try await model.prediction(from: provider)
        guard let waveform = out.featureValue(for: "waveform")?.multiArrayValue else {
            throw NSError(domain: "VibeVoiceKit.VAERunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "missing waveform output"])
        }
        // Expected shape (1, 1, 51200). Copy to [Float].
        let count = waveform.count
        var samples = [Float](repeating: 0, count: count)
        let stride = waveform.strides.map(\.intValue)
        // Dense row-major assumption — check stride(last)==1.
        precondition(stride.last == 1, "VAE output must be contiguous")
        let ptr = waveform.dataPointer.assumingMemoryBound(to: Float32.self)
        samples.withUnsafeMutableBufferPointer { dst in
            _ = dst.baseAddress.map { $0.update(from: ptr, count: count) }
        }
        return samples
    }

    /// Run a zero-input prediction through the VAE to amortize first-call
    /// CoreML compile/plan cost off the synthesis critical path.
    public func warmup() async throws {
        let latents = try MLMultiArray(shape: [1, 64, 16], dataType: .float32)
        memset(latents.dataPointer, 0, latents.count * MemoryLayout<Float>.size)
        _ = try await decode(latents: latents)
    }
}
