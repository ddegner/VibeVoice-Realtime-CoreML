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

/// Compile an `.mlpackage` to `.mlmodelc` and load it. `MLModel(contentsOf:)`
/// only accepts compiled models; Swift at runtime does not do the auto-compile
/// that Xcode does at build time.
///
/// Cache resolution (first hit wins):
///   1. `<package>.mlmodelc` next to the source — fastest, used when the
///      package directory is writable (dev tree, downloaded artifacts).
///   2. `<Caches>/VibeVoiceKit/<hash>/<name>.mlmodelc` — fallback used when the
///      package lives in a read-only location (e.g. inside `.app/Contents/
///      Resources/fp16/`). Hashed by the package's absolute path so two app
///      installs in different paths don't collide.
enum ModelLoader {
    static func load(packageURL: URL, configuration: MLModelConfiguration) async throws -> MLModel {
        let sibling = packageURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: sibling.path) {
            return try MLModel(contentsOf: sibling, configuration: configuration)
        }
        let cacheFallback = cacheFallbackURL(for: packageURL)
        if FileManager.default.fileExists(atPath: cacheFallback.path) {
            return try MLModel(contentsOf: cacheFallback, configuration: configuration)
        }

        let compiled = try await MLModel.compileModel(at: packageURL)
        // Try sibling first (fastest reload), fall back to Caches/ if the
        // package directory is read-only (e.g. shipped in an .app bundle).
        let dest: URL
        do {
            try? FileManager.default.removeItem(at: sibling)
            try FileManager.default.moveItem(at: compiled, to: sibling)
            dest = sibling
        } catch {
            try FileManager.default.createDirectory(
                at: cacheFallback.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: cacheFallback)
            try FileManager.default.moveItem(at: compiled, to: cacheFallback)
            dest = cacheFallback
        }
        return try MLModel(contentsOf: dest, configuration: configuration)
    }

    private static func cacheFallbackURL(for packageURL: URL) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let hash = String(packageURL.path.hashValue, radix: 16)
        let name = packageURL.deletingPathExtension().lastPathComponent + ".mlmodelc"
        return caches
            .appendingPathComponent("VibeVoiceKit", isDirectory: true)
            .appendingPathComponent(hash, isDirectory: true)
            .appendingPathComponent(name)
    }
}
