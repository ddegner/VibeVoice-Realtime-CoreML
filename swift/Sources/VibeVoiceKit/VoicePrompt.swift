import Foundation
import CoreML

/// Loads a pre-computed voice prompt (KV caches + last hidden state) exported
/// by `08_export_voice_prompt.py`. These caches encode the voice identity and
/// are required for the TTS LM to produce intelligible speech.
///
/// Each voice directory contains `.npy` files for four branches:
///   lm, tts_lm, neg_lm, neg_tts_lm
/// Each branch has: `*_hidden.npy`, `*_k_cache.npy`, `*_v_cache.npy`
/// Plus a `metadata.json` with sequence lengths.
public struct VoicePrompt {
    public struct Branch {
        public let hidden: MLMultiArray    // (1, T, 896)
        public let kCache: MLMultiArray    // (L, 1, KVH, T, D)
        public let vCache: MLMultiArray    // (L, 1, KVH, T, D)
        public let length: Int             // T
    }

    public let lm: Branch
    public let ttsLM: Branch
    public let negLM: Branch
    public let negTTSLM: Branch

    public init(voiceDir: URL) throws {
        struct Meta: Codable {
            let lm_len: Int
            let tts_lm_len: Int
            let neg_lm_len: Int
            let neg_tts_lm_len: Int
        }
        let metaData = try Data(contentsOf: voiceDir.appendingPathComponent("metadata.json"))
        let meta = try JSONDecoder().decode(Meta.self, from: metaData)

        self.lm = try Self.loadBranch(dir: voiceDir, prefix: "lm", length: meta.lm_len)
        self.ttsLM = try Self.loadBranch(dir: voiceDir, prefix: "tts_lm", length: meta.tts_lm_len)
        self.negLM = try Self.loadBranch(dir: voiceDir, prefix: "neg_lm", length: meta.neg_lm_len)
        self.negTTSLM = try Self.loadBranch(dir: voiceDir, prefix: "neg_tts_lm", length: meta.neg_tts_lm_len)
    }

    private static func loadBranch(dir: URL, prefix: String, length: Int) throws -> Branch {
        let h = try loadNpy(dir.appendingPathComponent("\(prefix)_hidden.npy"))
        let k = try loadNpy(dir.appendingPathComponent("\(prefix)_k_cache.npy"))
        let v = try loadNpy(dir.appendingPathComponent("\(prefix)_v_cache.npy"))
        return Branch(hidden: h, kCache: k, vCache: v, length: length)
    }

    /// Minimal .npy loader for float32 C-contiguous arrays.
    /// Numpy .npy format: 6-byte magic, 2-byte version, 2-byte header_len,
    /// ASCII header (Python dict with 'descr', 'fortran_order', 'shape'),
    /// then raw data.
    private static func loadNpy(_ url: URL) throws -> MLMultiArray {
        let data = try Data(contentsOf: url)
        // Parse header
        guard data.count > 10,
              data[0] == 0x93, // \x93
              data[1] == 0x4E, // N
              data[2] == 0x55, // U
              data[3] == 0x4D, // M
              data[4] == 0x50, // P
              data[5] == 0x59  // Y
        else {
            throw NSError(domain: "VoicePrompt", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "not a .npy file: \(url.lastPathComponent)"])
        }
        let major = data[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            // v2: 4-byte header length
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        }
        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else {
            throw NSError(domain: "VoicePrompt", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "truncated npy header"])
        }
        let headerStr = String(data: data[headerStart..<headerEnd], encoding: .ascii) ?? ""

        // Parse shape from header: 'shape': (d0, d1, ...)
        let shape = parseShape(headerStr)
        guard !shape.isEmpty else {
            throw NSError(domain: "VoicePrompt", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "could not parse shape from \(headerStr)"])
        }

        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .float32)
        let byteCount = shape.reduce(1, *) * MemoryLayout<Float>.size
        guard headerEnd + byteCount <= data.count else {
            throw NSError(domain: "VoicePrompt", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "truncated npy data"])
        }
        data.copyBytes(to: arr.dataPointer.assumingMemoryBound(to: UInt8.self),
                       from: headerEnd..<(headerEnd + byteCount))
        return arr
    }

    private static func parseShape(_ header: String) -> [Int] {
        // Find 'shape': (d0, d1, ...) in the numpy header dict string
        guard let range = header.range(of: "'shape':\\s*\\(([^)]+)\\)", options: .regularExpression) else {
            return []
        }
        let match = String(header[range])
        // Extract the tuple contents
        guard let parenStart = match.firstIndex(of: "("),
              let parenEnd = match.firstIndex(of: ")") else { return [] }
        let inner = match[match.index(after: parenStart)..<parenEnd]
        return inner.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
}
