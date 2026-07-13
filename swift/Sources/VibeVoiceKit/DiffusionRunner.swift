import Foundation
import CoreML

/// Runs the DPM-Solver++ (order 2, midpoint) v-prediction denoising loop in
/// Swift, calling the CoreML diffusion-head model 2× per step (CFG).
///
/// This matches upstream's `DPMSolverMultistepScheduler` configuration:
///   algorithm_type = "dpmsolver++"
///   solver_order = 2
///   solver_type = "midpoint"
///   lower_order_final = True
///   final_sigmas_type = "zero"
///   beta_schedule = "cosine"
///   prediction_type = "v_prediction"
///   timestep_spacing = "linspace"
public final class DiffusionRunner: @unchecked Sendable {
    /// Latent frames per VAE window.
    public static let T = 16
    /// Latent channel dim.
    public static let D = 64
    /// Conditioning hidden dim.
    public static let H = 896

    private let model: MLModel
    private let batchedModel: MLModel?
    /// DPM-Solver++ step count for each frame. Publicly readable so the
    /// timing reporter can log "N steps × 2 CFG" without hardcoding.
    public let steps: Int
    private let guidance: Float
    private let trainSteps: Int = 1000

    /// Seed copied into a fresh per-synthesis session. Keeping the mutable RNG
    /// and timestep buffer in that session lets independent chunks share the
    /// same immutable CoreML model safely.
    private let seedLock = NSLock()
    private var defaultSeed: UInt64? = nil

    // Pre-computed schedule
    private struct ScheduleEntry {
        let timestep: Int
        let alphaT: Double    // sqrt(alpha_bar_t)
        let sigmaT: Double    // sqrt(1 - alpha_bar_t)
        let lambdaT: Double   // log(alpha_t / sigma_t)
    }
    private let schedule: [ScheduleEntry]
    private let finalAlpha: Double  // alpha at sigma=0 (i.e., 1.0)
    private let finalSigma: Double  // 0.0

    public final class Session: @unchecked Sendable {
        fileprivate var rng: SplitMix64?
        fileprivate let timestep: MLMultiArray

        fileprivate init(seed: UInt64?) throws {
            self.rng = seed.map(SplitMix64.init(seed:))
            self.timestep = try MLMultiArray(shape: [1], dataType: .float32)
        }
    }

    public init(modelURL: URL, mlConfig: MLModelConfiguration, steps: Int, guidanceScale: Float) async throws {
        self.model = try await ModelLoader.load(packageURL: modelURL, configuration: mlConfig)
        let batchedURL = modelURL.deletingLastPathComponent()
            .appendingPathComponent("diffusion_head_cfg2.mlpackage")
        let batchedCompiled = batchedURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: batchedURL.path)
            || FileManager.default.fileExists(atPath: batchedCompiled.path) {
            self.batchedModel = try await ModelLoader.load(
                packageURL: batchedURL, configuration: mlConfig)
        } else {
            self.batchedModel = nil
        }
        self.steps = steps
        self.guidance = guidanceScale

        // Cosine beta schedule, matching HF `betas_for_alpha_bar(alpha_transform_type="cosine")`
        // followed by cumprod(1 - beta). Earlier versions of this file
        // (mis)computed alpha_cumprod[t] = alpha_bar_fn(t/N) / alpha_bar_fn(0),
        // which is off-by-one relative to HF's telescoping-ratio form and
        // gives alpha(t=999)=3e-3 (clamped) instead of the correct ~5e-5 once
        // the max_beta=0.999 clip at i=999 kicks in. That 65× error on the
        // FIRST diffusion step changes lambda by ~4 and feeds slightly-off
        // x0 predictions to the acoustic feedback loop.
        let s = 0.008
        let maxBeta = 0.999
        let N = 1000
        var alphaBars = [Double](repeating: 0, count: N)
        var cumprod = 1.0
        func alphaBarFn(_ u: Double) -> Double {
            pow(cos((u + s) / (1.0 + s) * .pi / 2.0), 2)
        }
        for t in 0..<N {
            let t1 = Double(t) / Double(N)
            let t2 = Double(t + 1) / Double(N)
            let beta = min(1.0 - alphaBarFn(t2) / alphaBarFn(t1), maxBeta)
            cumprod *= (1.0 - beta)
            alphaBars[t] = cumprod
        }

        // Linspace timestep spacing (matching upstream)
        // np.linspace(0, 999, steps+1).round()[::-1][:-1]
        var timesteps = [Int]()
        timesteps.reserveCapacity(steps)
        for i in 0..<(steps + 1) {
            let val = Double(i) * 999.0 / Double(steps)
            timesteps.append(Int(val.rounded()))
        }
        timesteps.reverse()
        timesteps.removeLast() // drop the 0 at the end

        // Build schedule entries
        var sched = [ScheduleEntry]()
        sched.reserveCapacity(steps)
        for t in timesteps {
            let aBar = alphaBars[t]
            let alphaT = aBar.squareRoot()
            let sigmaT = (1.0 - aBar).squareRoot()
            let lambdaT = log(alphaT / sigmaT)
            sched.append(ScheduleEntry(timestep: t, alphaT: alphaT, sigmaT: sigmaT, lambdaT: lambdaT))
        }
        self.schedule = sched
        self.finalAlpha = 1.0   // final_sigmas_type = "zero"
        self.finalSigma = 0.0

    }

    // MARK: - Public API

    /// Denoise a single speech latent frame using DPM-Solver++ order 2.
    /// Returns shape (1, 1, 64).
    public func makeSession() throws -> Session {
        seedLock.lock()
        let seed = defaultSeed
        seedLock.unlock()
        return try Session(seed: seed)
    }

    public func denoiseFrame(
        cond: MLMultiArray,
        uncond: MLMultiArray,
        session: Session
    ) async throws -> MLMultiArray {
        try Task.checkCancellation()
        let cond3 = try Self.expandToT1(cond, H: Self.H)
        let uncond3 = try Self.expandToT1(uncond, H: Self.H)

        var sample = try randomNormal(shape: [1, 1, Self.D] as [NSNumber], session: session)
        let tPtr = session.timestep.dataPointer.assumingMemoryBound(to: Float.self)

        // DPM-Solver++ state: history of x0 predictions for multistep
        var prevX0: MLMultiArray? = nil
        var orderUsed = 0

        for i in 0..<steps {
            try Task.checkCancellation()
            let entry = schedule[i]
            tPtr[0] = Float(entry.timestep)

            // CFG branches are independent — dispatch both concurrently.
            // `sample` and the session timestep are read-only for the duration
            // of these two calls (we only write to `tPtr[0]` *before* dispatch),
            // and the CoreML model is thread-safe. The `let sampleIn` binding
            // satisfies Swift 6's rule that `async let` captures must be
            // `Sendable` / `let` — the MLMultiArray reference itself is
            // copied, not its memory.
            let sampleIn = sample
            let vCond: MLMultiArray
            let vUncond: MLMultiArray
            if let batchedModel {
                let pair = try await predictCFG(
                    model: batchedModel,
                    latent: sampleIn,
                    t: session.timestep,
                    cond: cond3,
                    uncond: uncond3)
                vCond = pair.cond
                vUncond = pair.uncond
            } else {
                async let vCondTask = predict(latent: sampleIn, t: session.timestep, cond: cond3)
                async let vUncondTask = predict(latent: sampleIn, t: session.timestep, cond: uncond3)
                (vCond, vUncond) = try await (vCondTask, vUncondTask)
            }
            try Task.checkCancellation()
            let v = try cfgBlend(uncond: vUncond, cond: vCond, scale: guidance)

            // v-prediction → x0
            let x0 = try vPredToX0(sample: sample, v: v, alphaT: entry.alphaT, sigmaT: entry.sigmaT)

            // Determine target sigma/alpha for this step
            let targetAlpha: Double
            let targetSigma: Double
            let targetLambda: Double
            if i + 1 < steps {
                targetAlpha = schedule[i + 1].alphaT
                targetSigma = schedule[i + 1].sigmaT
                targetLambda = schedule[i + 1].lambdaT
            } else {
                // Last step → final sigma = 0
                targetAlpha = finalAlpha
                targetSigma = finalSigma
                targetLambda = .infinity  // not used in first-order
            }

            // Determine order: first step and last step use order 1 (lower_order_final)
            let thisOrder: Int
            if i == 0 || i == steps - 1 {
                thisOrder = 1
            } else {
                thisOrder = min(2, orderUsed + 1)
            }

            if thisOrder == 1 {
                // First-order DPM-Solver++ update
                let h = targetLambda - entry.lambdaT
                sample = try dpmSolverFirstOrder(
                    sample: sample, x0: x0,
                    sigmaS: entry.sigmaT, alphaT: targetAlpha, sigmaT: targetSigma, h: h)
            } else {
                // Second-order multistep (midpoint) update
                // h = lambda_target - lambda_current
                // r0 = (lambda_current - lambda_previous) / h
                let h = targetLambda - entry.lambdaT
                let r0 = (entry.lambdaT - schedule[i - 1].lambdaT) / h
                guard let prev = prevX0 else {
                    throw NSError(domain: "VibeVoiceKit.DiffusionRunner", code: 3, userInfo: [
                        NSLocalizedDescriptionKey: "order-2 step at i=\(i) without prevX0 — order progression invariant violated"
                    ])
                }
                sample = try dpmSolverSecondOrder(
                    sample: sample, x0: x0, prevX0: prev,
                    sigmaS: entry.sigmaT, alphaT: targetAlpha, sigmaT: targetSigma,
                    h: h, r0: r0)
            }

            prevX0 = x0
            orderUsed = thisOrder
        }

        return sample
    }

    // MARK: - DPM-Solver++ step functions

    /// First-order DPM-Solver++ update:
    ///   x_t = (sigma_t / sigma_s) * x_s + alpha_t * (1 - exp(-h)) * x0
    private func dpmSolverFirstOrder(
        sample: MLMultiArray, x0: MLMultiArray,
        sigmaS: Double, alphaT: Double, sigmaT: Double, h: Double
    ) throws -> MLMultiArray {
        let n = sample.count
        let out = try MLMultiArray(shape: sample.shape, dataType: .float32)
        let xs = sample.dataPointer.assumingMemoryBound(to: Float.self)
        let x0p = x0.dataPointer.assumingMemoryBound(to: Float.self)
        let op = out.dataPointer.assumingMemoryBound(to: Float.self)

        // For the last step: sigmaT = 0, so the ratio term vanishes
        // and x_t = alpha_t * x0 (clean prediction)
        if sigmaT < 1e-12 {
            let a = Float(alphaT)
            for i in 0..<n { op[i] = a * x0p[i] }
            return out
        }

        let ratio = Float(sigmaT / sigmaS)
        let coeff = Float(alphaT * (1.0 - exp(-h)))
        for i in 0..<n {
            op[i] = ratio * xs[i] + coeff * x0p[i]
        }
        return out
    }

    /// Second-order DPM-Solver++ multistep (midpoint) update:
    ///   D0 = x0_current
    ///   D1 = (1/r0) * (x0_current - x0_previous)
    ///   x_t = (sigma_t/sigma_s) * x_s + alpha_t * (exp(-h)-1) * D0
    ///         + 0.5 * alpha_t * (exp(-h)-1) * D1
    private func dpmSolverSecondOrder(
        sample: MLMultiArray, x0: MLMultiArray, prevX0: MLMultiArray,
        sigmaS: Double, alphaT: Double, sigmaT: Double,
        h: Double, r0: Double
    ) throws -> MLMultiArray {
        let n = sample.count
        let out = try MLMultiArray(shape: sample.shape, dataType: .float32)
        let xs = sample.dataPointer.assumingMemoryBound(to: Float.self)
        let x0p = x0.dataPointer.assumingMemoryBound(to: Float.self)
        let px0p = prevX0.dataPointer.assumingMemoryBound(to: Float.self)
        let op = out.dataPointer.assumingMemoryBound(to: Float.self)

        if sigmaT < 1e-12 {
            let a = Float(alphaT)
            for i in 0..<n { op[i] = a * x0p[i] }
            return out
        }

        let ratio = Float(sigmaT / sigmaS)
        let expmh_m1 = Float(exp(-h) - 1.0)  // negative
        let aCoeff = Float(alphaT) * expmh_m1
        let invR0 = Float(1.0 / r0)

        for i in 0..<n {
            let d0 = x0p[i]
            let d1 = invR0 * (x0p[i] - px0p[i])
            op[i] = ratio * xs[i] - aCoeff * d0 - 0.5 * aCoeff * d1
        }
        return out
    }

    // MARK: - Helpers

    /// v-prediction → x0: x0 = alpha_t * x_t - sigma_t * v
    private func vPredToX0(sample: MLMultiArray, v: MLMultiArray,
                           alphaT: Double, sigmaT: Double) throws -> MLMultiArray {
        let n = sample.count
        let out = try MLMultiArray(shape: sample.shape, dataType: .float32)
        let x = sample.dataPointer.assumingMemoryBound(to: Float.self)
        let vp = v.dataPointer.assumingMemoryBound(to: Float.self)
        let op = out.dataPointer.assumingMemoryBound(to: Float.self)
        let a = Float(alphaT)
        let s = Float(sigmaT)
        for i in 0..<n { op[i] = a * x[i] - s * vp[i] }
        return out
    }

    private func cfgBlend(uncond: MLMultiArray, cond: MLMultiArray, scale: Float) throws -> MLMultiArray {
        let n = uncond.count
        let out = try MLMultiArray(shape: uncond.shape, dataType: .float32)
        let u = uncond.dataPointer.assumingMemoryBound(to: Float.self)
        let c = cond.dataPointer.assumingMemoryBound(to: Float.self)
        let o = out.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<n { o[i] = u[i] + scale * (c[i] - u[i]) }
        return out
    }

    private func predict(latent: MLMultiArray, t: MLMultiArray, cond: MLMultiArray) async throws -> MLMultiArray {
        try Task.checkCancellation()
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "noisy_latent": latent, "timestep": t, "conditioning": cond,
        ])
        let out = try await model.prediction(from: provider)
        try Task.checkCancellation()
        guard let v = out.featureValue(for: "predicted_noise")?.multiArrayValue else {
            throw NSError(domain: "VibeVoiceKit.DiffusionRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing predicted_noise"])
        }
        return v
    }

    private func predictCFG(
        model: MLModel,
        latent: MLMultiArray,
        t: MLMultiArray,
        cond: MLMultiArray,
        uncond: MLMultiArray
    ) async throws -> (cond: MLMultiArray, uncond: MLMultiArray) {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "noisy_latent": latent,
            "timestep": t,
            "conditioning": cond,
            "unconditioning": uncond,
        ])
        let output = try await model.prediction(from: provider)
        guard
            let condOut = output.featureValue(for: "predicted_noise_cond")?.multiArrayValue,
            let uncondOut = output.featureValue(for: "predicted_noise_uncond")?.multiArrayValue
        else {
            throw NSError(domain: "VibeVoiceKit.DiffusionRunner", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "batched diffusion outputs missing"])
        }
        return (condOut, uncondOut)
    }

    static func expandToT1(_ a: MLMultiArray, H: Int) throws -> MLMultiArray {
        if a.shape.count == 3 { return a }
        precondition(a.shape.map(\.intValue) == [1, H], "unexpected cond shape \(a.shape)")
        let out = try MLMultiArray(shape: [1, 1, NSNumber(value: H)], dataType: .float32)
        memcpy(out.dataPointer, a.dataPointer, H * MemoryLayout<Float>.size)
        return out
    }

    private func randomNormal(shape: [NSNumber], session: Session) throws -> MLMultiArray {
        let out = try MLMultiArray(shape: shape, dataType: .float32)
        let p = out.dataPointer.assumingMemoryBound(to: Float.self)
        // Box-Muller in Double precision (truncate to Float only at store time).
        // Doing Box-Muller in Float introduces ~1e-7 relative error per noise
        // sample vs the fp64 reference path used by PyTorch and by script 11's
        // diagnostic port. Because the diffusion loop runs the prediction head
        // 2×CFG × steps × N frames times, that small noise difference compounds
        // and — on certain seeds — pushes the trajectory onto a chaotic branch
        // where the TTS-LM picks a different next token, producing word-repeat
        // stutters ("lazy" → "la-lazy"). Keep the uniform draws and Box-Muller
        // arithmetic in fp64 so the noise stream matches the reference bit-for-
        // bit (within Double roundoff).
        if session.rng != nil {
            for i in 0..<out.count {
                let u1 = max(Double.leastNormalMagnitude, session.rng!.nextUnit())
                let u2 = session.rng!.nextUnit()
                let n = (-2.0 * log(u1)).squareRoot() * cos(2.0 * .pi * u2)
                p[i] = Float(n)
            }
        } else {
            for i in 0..<out.count {
                let u1 = max(Double.leastNormalMagnitude, Double.random(in: 0..<1))
                let u2 = Double.random(in: 0..<1)
                let n = (-2.0 * log(u1)).squareRoot() * cos(2.0 * .pi * u2)
                p[i] = Float(n)
            }
        }
        return out
    }

    /// Seed the diffusion noise RNG for reproducible output. Pass `nil` to
    /// revert to the system (non-deterministic) RNG.
    public func setSeed(_ seed: UInt64?) {
        seedLock.lock()
        defaultSeed = seed
        seedLock.unlock()
    }
}

/// Tiny SplitMix64 PRNG — pure state transitions, reproducible across runs,
/// adequate distribution for diffusion noise. Single-threaded only.
fileprivate struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform Double in [0, 1).
    mutating func nextUnit() -> Double {
        // Top 53 bits → IEEE double mantissa.
        return Double(next() >> 11) * (1.0 / Double(1 << 53))
    }
}
