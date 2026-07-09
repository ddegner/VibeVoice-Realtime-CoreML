import Foundation
import CoreML

/// Drives the text LM and TTS LM autoregressive decode loops.
///
/// Voice prompts ship pre-computed KV caches (built externally by
/// `08_export_voice_prompt.py`), so the prefill graphs are unused at runtime —
/// `TTSLoop` feeds tokens directly through the decode models, extending the
/// voice prompt's caches one position at a time.
public final class LLMRunner: @unchecked Sendable {
    private let textDecode: MLModel
    /// One or more handles to the same `tts_lm_decode.mlpackage`. Multiple
    /// handles let the pos/neg CFG branches dispatch through independent
    /// MLModel instances; whether this actually unlocks GPU concurrency vs
    /// a single handle is a per-platform measurement question (Phase 5
    /// sweep). Defaults to 1.
    private let ttsDecodes: [MLModel]

    // NOTE: decode input arrays are allocated **per call** rather than kept as
    // stored properties. Earlier revisions pre-allocated (1,1) buffers on `self`
    // and mutated them in-place inside each step to save allocation cost, but
    // that made `ttsDecodeStep` unsafe to call concurrently for the pos/neg
    // CFG branches — the two calls would race on the same scratch memory.
    // The per-call allocation is negligible (4 tiny MLMultiArrays, ~microseconds)
    // compared to the 15-20ms CoreML forward, and it unlocks ~15% total
    // speedup by letting TTSLoop do `async let posStep / async let negStep`.

    public convenience init(modelsDir: URL, config: MLModelConfiguration) async throws {
        try await self.init(
            modelsDir: modelsDir, textConfig: config, ttsConfig: config)
    }

    /// Per-model config variant. Text-LM and TTS-LM may be routed to
    /// different compute units (e.g. ANE vs GPU) for the Phase 3 sweep.
    /// `ttsHandles` controls Phase 5 CFG multiplex (default 1).
    public init(
        modelsDir: URL,
        textConfig: MLModelConfiguration,
        ttsConfig: MLModelConfiguration,
        ttsHandles: Int = 1
    ) async throws {
        func load(_ name: String, config: MLModelConfiguration, required: Bool) async throws -> MLModel? {
            let url = modelsDir.appendingPathComponent(name)
            // Accept either the source `.mlpackage` (which ModelLoader will
            // compile + cache) or a pre-compiled `.mlmodelc` sibling. The
            // latter lets us ship `.mlmodelc`-only deployments (e.g. bundled
            // inside an app) without carrying the unused `.mlpackage` source.
            let mlmodelc = url.deletingPathExtension().appendingPathExtension("mlmodelc")
            let fm = FileManager.default
            guard fm.fileExists(atPath: url.path) || fm.fileExists(atPath: mlmodelc.path) else {
                if required {
                    throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
                }
                return nil
            }
            return try await ModelLoader.load(packageURL: url, configuration: config)
        }
        self.textDecode = try await load("text_lm_decode.mlpackage", config: textConfig, required: true)!
        var handles: [MLModel] = []
        let count = max(1, ttsHandles)
        for _ in 0..<count {
            handles.append(try await load("tts_lm_decode.mlpackage", config: ttsConfig, required: true)!)
        }
        self.ttsDecodes = handles
    }

    // MARK: - Per-call scratch allocation (concurrency-safe)

    private static func makeIDs(_ value: Int32) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, 1], dataType: .int32)
        a.dataPointer.assumingMemoryBound(to: Int32.self)[0] = value
        return a
    }
    private static func makeTail(_ value: Float) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        a.dataPointer.assumingMemoryBound(to: Float.self)[0] = value
        return a
    }

    /// One *text* LM decode step (plain Qwen2 graph).
    public func textDecodeStep(
        inputID: Int32,
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        try Task.checkCancellation()
        let ids = try Self.makeIDs(inputID)
        let pos = try Self.makeIDs(positionID)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids, "k_cache": kCache, "v_cache": vCache, "position_ids": pos,
        ])
        let out = try await textDecode.prediction(from: provider)
        try Task.checkCancellation()
        return try Self.unpackDecode(out)
    }

    /// One *TTS* LM decode step. Matches the mask-form graph from
    /// `02b_convert_tts_lm.py` (TTSDecodeWrapper):
    ///
    ///   inputs_embeds = tok_embed * (1 - tail_mask) + lm_hidden * tail_mask
    ///                 + tts_input_types(tts_text_masks)
    ///
    /// - Parameters:
    ///   - inputID: token id. At speech-frame steps upstream passes 1 as a
    ///       sentinel; the embedding gets overwritten by `lmHidden` anyway
    ///       when `tailMask == 1`.
    ///   - lmHidden: (1, 1, 896) — text LM hidden (at text step) OR
    ///       `acoustic_connector(latent)` output (at speech step).
    ///   - tailMask: 1.0 if the slot should be overridden by `lmHidden`,
    ///       0.0 if the plain token embedding should pass through. Upstream
    ///       only overrides the tail, but at decode the step *is* the tail.
    ///   - textMask: 1 = text position, 0 = speech position. Drives
    ///       `tts_input_types` embedding.
    public func ttsDecodeStep(
        branch: Int = 0,
        inputID: Int32,
        lmHidden: MLMultiArray,
        tailMask: Float,
        textMask: Int32,
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        try Task.checkCancellation()
        // Per-call scratch — required so two concurrent calls (pos/neg CFG
        // branches) don't clobber each other's inputs mid-flight.
        let ids = try Self.makeIDs(inputID)
        let tail = try Self.makeTail(tailMask)
        let tm = try Self.makeIDs(textMask)
        let pos = try Self.makeIDs(positionID)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids,
            "lm_hidden_padded": lmHidden,
            "tail_mask": tail,
            "tts_text_masks": tm,
            "k_cache": kCache,
            "v_cache": vCache,
            "position_ids": pos,
        ])
        let model = ttsDecodes[branch % ttsDecodes.count]
        let out = try await model.prediction(from: provider)
        try Task.checkCancellation()
        return try Self.unpackDecode(out)
    }

    private static func unpackDecode(_ out: MLFeatureProvider) throws
        -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray)
    {
        guard
            let h = out.featureValue(for: "hidden_states")?.multiArrayValue,
            let kNew = out.featureValue(for: "k_cache_out")?.multiArrayValue,
            let vNew = out.featureValue(for: "v_cache_out")?.multiArrayValue
        else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "decode outputs missing"])
        }
        return (h, kNew, vNew)
    }

}
