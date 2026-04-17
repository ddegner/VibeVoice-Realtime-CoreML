import Foundation
import AVFoundation

/// Thin wrapper around AVAudioEngine for 24 kHz mono float PCM playback.
/// Accepts chunks so streaming mode (Phase G) can feed buffers as they arrive.
public final class AudioPlayer {
    public static let sampleRate: Double = 24_000

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat

    public init() throws {
        guard let fmt = AVAudioFormat(
            standardFormatWithSampleRate: Self.sampleRate,
            channels: 1
        ) else {
            throw NSError(domain: "VibeVoiceKit.AudioPlayer", code: -1)
        }
        self.format = fmt
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: fmt)
        try engine.start()
    }

    public func play(_ samples: [Float]) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channel = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    channel.update(from: base, count: samples.count)
                }
            }
        }
        player.scheduleBuffer(buffer, completionHandler: nil)
        if !player.isPlaying { player.play() }
    }

    public func stop() {
        player.stop()
    }

    /// Write a mono float buffer to a 32-bit IEEE-float WAV file at 24 kHz.
    public static func writeWAV(_ samples: [Float], to url: URL) throws {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { throw NSError(domain: "VibeVoiceKit.AudioPlayer", code: -2) }

        let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw NSError(domain: "VibeVoiceKit.AudioPlayer", code: -3)
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    ch.update(from: base, count: samples.count)
                }
            }
        }
        try file.write(from: buffer)
    }
}
