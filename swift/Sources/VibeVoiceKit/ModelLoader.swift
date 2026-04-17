import Foundation
import CoreML

// MARK: - Retroactive Sendable conformance
//
// `MLMultiArray` and `MLModel` are not declared `Sendable` by Apple, but we
// pass them across `async let` boundaries (pos/neg CFG branches, cond/uncond
// diffusion predictions). For our use these are safe because:
//  - `MLMultiArray` inputs handed to `model.prediction(from:)` are treated as
//    read-only by CoreML (Apple's documented contract). Our code does not
//    mutate an array after spawning an async task that captures it and before
//    awaiting the task — the schedule is "write, spawn tasks, await".
//  - `MLModel.prediction(from:)` is thread-safe (re-entrant) — the same model
//    handle can satisfy concurrent prediction requests on different input
//    tensors. This is exercised throughout CoreML's sample code.
//
// Removing these conformances would force us back to sequential dispatch and
// give up the ~15-30% throughput gain from overlapping the two independent
// CFG passes and the two CFG LM branches.
extension MLMultiArray: @unchecked @retroactive Sendable {}
extension MLModel: @unchecked @retroactive Sendable {}

/// Compile an `.mlpackage` to `.mlmodelc` (cached next to the package) and
/// load it. `MLModel(contentsOf:)` only accepts compiled models; Swift at
/// runtime does not do the auto-compile that Xcode does at build time.
enum ModelLoader {
    static func load(packageURL: URL, configuration: MLModelConfiguration) async throws -> MLModel {
        let cachedCompiled = packageURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: cachedCompiled.path) {
            return try MLModel(contentsOf: cachedCompiled, configuration: configuration)
        }
        let compiled = try await MLModel.compileModel(at: packageURL)
        // Move into place so subsequent loads skip recompilation.
        try? FileManager.default.removeItem(at: cachedCompiled)
        try FileManager.default.moveItem(at: compiled, to: cachedCompiled)
        return try MLModel(contentsOf: cachedCompiled, configuration: configuration)
    }
}
