import XCTest
@testable import VibeVoiceKit

final class SmokeTests: XCTestCase {
    func testAudioPlayerInit() throws {
        // Most minimal check the library links and AVAudioEngine config is sane.
        _ = try AudioPlayer()
    }
}
