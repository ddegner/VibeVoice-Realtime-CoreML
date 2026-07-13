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
    private let ttsCFGDecode: MLModel?
    private let textBlockDecode: MLModel?
    private let ttsTextBlockDecode: MLModel?

    public var hasBatchedCFG: Bool { ttsCFGDecode != nil }
    public var hasBlockDecode: Bool { textBlockDecode != nil && ttsTextBlockDecode != nil }

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
        self.ttsCFGDecode = try await load(
            "tts_lm_decode_cfg2.mlpackage", config: ttsConfig, required: false)
        self.textBlockDecode = try await load(
            "text_lm_decode_block5.mlpackage", config: textConfig, required: false)
        self.ttsTextBlockDecode = try await load(
            "tts_lm_decode_block5.mlpackage", config: ttsConfig, required: false)
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

    private static func makeIDs(_ values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for (index, value) in values.enumerated() { pointer[index] = value }
        return array
    }

    private static func makePositions(start: Int32, count: Int) throws -> MLMultiArray {
        try makeIDs((0..<count).map { start + Int32($0) })
    }

    private static func makeCausalMask(pastLength: Int, queryLength: Int) throws -> MLMultiArray {
        let totalLength = pastLength + queryLength
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: queryLength), NSNumber(value: totalLength)],
            dataType: .float32)
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for query in 0..<queryLength {
            let allowed = pastLength + query + 1
            let row = query * totalLength
            for key in 0..<totalLength {
                pointer[row + key] = key < allowed ? 0 : -10_000
            }
        }
        return array
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

    public func textDecodeBlock(
        inputIDs: [Int32],
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        guard inputIDs.count == 5, let model = textBlockDecode else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "text block model unavailable"])
        }
        let pastLength = kCache.shape[3].intValue
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": try Self.makeIDs(inputIDs),
            "k_cache": kCache,
            "v_cache": vCache,
            "position_ids": try Self.makePositions(start: positionID, count: inputIDs.count),
            "attention_mask": try Self.makeCausalMask(
                pastLength: pastLength, queryLength: inputIDs.count),
        ])
        let output = try await model.prediction(from: provider)
        return try Self.unpackDecode(output)
    }

    public func ttsTextDecodeBlock(
        lmHidden: MLMultiArray,
        kCache: MLMultiArray,
        vCache: MLMultiArray,
        positionID: Int32
    ) async throws -> (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray) {
        guard lmHidden.shape[1].intValue == 5, let model = ttsTextBlockDecode else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 7,
                          userInfo: [NSLocalizedDescriptionKey: "TTS text block model unavailable"])
        }
        let queryLength = lmHidden.shape[1].intValue
        let pastLength = kCache.shape[3].intValue
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "lm_hidden": lmHidden,
            "k_cache": kCache,
            "v_cache": vCache,
            "position_ids": try Self.makePositions(start: positionID, count: queryLength),
            "attention_mask": try Self.makeCausalMask(
                pastLength: pastLength, queryLength: queryLength),
        ])
        let output = try await model.prediction(from: provider)
        return try Self.unpackDecode(output)
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

    /// Speech-only CFG decode using one batch-2 CoreML prediction. Positive
    /// and negative voice caches have different logical lengths, so the
    /// shorter cache is right-aligned in a masked batch tensor and restored
    /// to its original logical length after prediction.
    public func ttsDecodeCFGStep(
        lmHidden: MLMultiArray,
        posK: MLMultiArray,
        posV: MLMultiArray,
        posPosition: Int32,
        negK: MLMultiArray,
        negV: MLMultiArray,
        negPosition: Int32
    ) async throws -> (
        pos: (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray),
        neg: (hidden: MLMultiArray, k: MLMultiArray, v: MLMultiArray)
    ) {
        guard let model = ttsCFGDecode else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "batched CFG model not loaded"])
        }
        try Task.checkCancellation()

        let posShape = posK.shape.map(\.intValue)
        let negShape = negK.shape.map(\.intValue)
        precondition(posShape.count == 5 && negShape.count == 5)
        precondition(posShape[0] == negShape[0] && posShape[1] == 1 && negShape[1] == 1)
        precondition(posShape[2] == negShape[2] && posShape[4] == negShape[4])
        let layers = posShape[0]
        let kvHeads = posShape[2]
        let dim = posShape[4]
        let posLength = posShape[3]
        let negLength = negShape[3]
        let commonLength = max(posLength, negLength)
        let posPad = commonLength - posLength
        let negPad = commonLength - negLength

        func pack(_ pos: MLMultiArray, _ neg: MLMultiArray) throws -> MLMultiArray {
            let out = try MLMultiArray(
                shape: [layers, 2, kvHeads, commonLength, dim].map(NSNumber.init(value:)),
                dataType: .float32)
            memset(out.dataPointer, 0, out.count * MemoryLayout<Float>.size)
            let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
            let posSrc = pos.dataPointer.assumingMemoryBound(to: Float.self)
            let negSrc = neg.dataPointer.assumingMemoryBound(to: Float.self)
            for layer in 0..<layers {
                for head in 0..<kvHeads {
                    let posSourceOffset = (layer * kvHeads + head) * posLength * dim
                    let posDestOffset = ((layer * 2) * kvHeads + head) * commonLength * dim
                        + posPad * dim
                    memcpy(dst.advanced(by: posDestOffset),
                           posSrc.advanced(by: posSourceOffset),
                           posLength * dim * MemoryLayout<Float>.size)

                    let negSourceOffset = (layer * kvHeads + head) * negLength * dim
                    let negDestOffset = ((layer * 2 + 1) * kvHeads + head) * commonLength * dim
                        + negPad * dim
                    memcpy(dst.advanced(by: negDestOffset),
                           negSrc.advanced(by: negSourceOffset),
                           negLength * dim * MemoryLayout<Float>.size)
                }
            }
            return out
        }

        let batchK = try pack(posK, negK)
        let batchV = try pack(posV, negV)
        let positions = try MLMultiArray(shape: [2, 1], dataType: .int32)
        let positionPtr = positions.dataPointer.assumingMemoryBound(to: Int32.self)
        positionPtr[0] = posPosition
        positionPtr[1] = negPosition

        let attentionMask = try MLMultiArray(
            shape: [2, 1, 1, commonLength + 1].map(NSNumber.init(value:)),
            dataType: .float32)
        let mask = attentionMask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(2 * (commonLength + 1)) { mask[i] = 0 }
        for i in 0..<posPad { mask[i] = -10_000 }
        let negMaskOffset = commonLength + 1
        for i in 0..<negPad { mask[negMaskOffset + i] = -10_000 }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "lm_hidden": lmHidden,
            "k_cache": batchK,
            "v_cache": batchV,
            "position_ids": positions,
            "attention_mask": attentionMask,
        ])
        let output = try await model.prediction(from: provider)
        try Task.checkCancellation()
        guard
            let hidden = output.featureValue(for: "hidden_states")?.multiArrayValue,
            let kOut = output.featureValue(for: "k_cache_out")?.multiArrayValue,
            let vOut = output.featureValue(for: "v_cache_out")?.multiArrayValue
        else {
            throw NSError(domain: "VibeVoiceKit.LLMRunner", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "batched CFG outputs missing"])
        }

        func splitHidden(branch: Int) throws -> MLMultiArray {
            let out = try MLMultiArray(shape: [1, 1, 896], dataType: .float32)
            let src = hidden.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(out.dataPointer, src.advanced(by: branch * 896),
                   896 * MemoryLayout<Float>.size)
            return out
        }

        func splitCache(
            _ batch: MLMultiArray,
            branch: Int,
            logicalLength: Int,
            pad: Int
        ) throws -> MLMultiArray {
            let outputLength = logicalLength + 1
            let batchLength = commonLength + 1
            let out = try MLMultiArray(
                shape: [layers, 1, kvHeads, outputLength, dim].map(NSNumber.init(value:)),
                dataType: .float32)
            let src = batch.dataPointer.assumingMemoryBound(to: Float.self)
            let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
            for layer in 0..<layers {
                for head in 0..<kvHeads {
                    let sourceOffset = ((layer * 2 + branch) * kvHeads + head)
                        * batchLength * dim + pad * dim
                    let destinationOffset = (layer * kvHeads + head) * outputLength * dim
                    memcpy(dst.advanced(by: destinationOffset),
                           src.advanced(by: sourceOffset),
                           outputLength * dim * MemoryLayout<Float>.size)
                }
            }
            return out
        }

        return (
            pos: (
                try splitHidden(branch: 0),
                try splitCache(kOut, branch: 0, logicalLength: posLength, pad: posPad),
                try splitCache(vOut, branch: 0, logicalLength: posLength, pad: posPad)),
            neg: (
                try splitHidden(branch: 1),
                try splitCache(kOut, branch: 1, logicalLength: negLength, pad: negPad),
                try splitCache(vOut, branch: 1, logicalLength: negLength, pad: negPad))
        )
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
