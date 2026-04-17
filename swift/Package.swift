// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VibeVoiceCoreML",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VibeVoiceKit", targets: ["VibeVoiceKit"]),
        .executable(name: "vibevoice-cli", targets: ["vibevoice-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
        .package(url: "https://github.com/apple/swift-argument-parser", "1.4.0" ..< "1.5.0"),
    ],
    targets: [
        .target(
            name: "VibeVoiceKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "vibevoice-cli",
            dependencies: [
                "VibeVoiceKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "VibeVoiceKitTests",
            dependencies: ["VibeVoiceKit"]
        ),
    ]
)
