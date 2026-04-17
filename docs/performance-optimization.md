# Performance optimization — decisions and evidence

> Companion to `conversion-notes.md`. That file is a chronological log;
> this one is opinionated. It captures *why* the current pipeline looks
> the way it does after performance work, so future optimization
> attempts don't re-tread the same dead ends.
>
> Written 2026-04-16 after the parallelization + fp16 + stateful-KV
> investigation. If you extend this: append, don't rewrite.

---

## TL;DR

The pipeline runs at **RTF 0.36–0.65** on an M-series dev laptop
(2.8×–1.5× faster than real-time). The path that got us there:

1. **Parallelized pos/neg TTS-LM decode and cond/uncond diffusion CFG
   with `async let`.** Mean RTF −32 % across a 23-clip regression
   matrix. 20/23 clips bit-identical to the sequential baseline; the
   other 3 differ only in the last 10 ms (tail-trim fade region, not a
   parallelism bug). **This is the whole story of the speedup.**

2. **Two other ideas that sounded good and weren't**:
   - **Stateful KV cache** (`ct.StateType`): measured shuttle cost is
     0.2 % of step time, not the 20–30 % I'd estimated. Upside after
     a full rewrite is ~2 %. Don't do it.
   - **Post-training fp16** (`linear_quantize_weights`): 21× *slower*,
     not faster. Quantizes weights, leaves compute at fp32 with a
     per-op upcast; VAE in particular falls to CPU. The
     `conversion/out/fp16/` directory is smaller for distribution
     but slower for execution; never point the CLI at it.

3. **What would actually move the needle next**: a proper
   `compute_precision=FLOAT16` re-export. Real GPU fp16 arithmetic.
   Non-trivial (per-model parity, VAE overflow risk). Deferred.

Everything below is the evidence for those three points, plus the
Swift 6 concurrency gotchas hit along the way and the verification
methodology so future regressions are easy to catch.

---

## The speedup that worked: `async let` parallelization

### Why this was available

Per speech frame, VibeVoice does two kinds of inherently-parallel
work:

**Pos/neg TTS-LM decode for CFG.** The TTS LM runs twice per frame
— once on the positive (voice-conditioned) KV cache, once on the
negative (uncond) KV cache — to produce the two hidden vectors the
diffusion head needs. The two calls share nothing but the
read-only `acoustic_connector(latent)` input and the `llm` model
reference.

**Cond/uncond diffusion CFG.** Inside each DPM-Solver++ step, the
denoiser runs twice — once on the positive conditioning, once on
the negative — to produce `v_cond` and `v_uncond`, which are then
blended by the guidance scale. Same shape of independence.

Both were sequential. `CoreML` `MLModel.prediction(from:)` is
documented re-entrant and the inputs are not mutated by the
forward pass, so concurrent calls are safe as long as each call
owns its own input `MLMultiArray` scratch.

### The change

Per-frame TTS-LM, in `swift/Sources/VibeVoiceKit/TTSLoop.swift`:

```swift
// before:
let posStep = try await llm.ttsDecodeStep(...)
let negStep = try await llm.ttsDecodeStep(...)

// after:
let posKIn = posK, posVIn = posV, posPosIn = posPos
let negKIn = negK, negVIn = negV, negPosIn = negPos
async let posTask = llm.ttsDecodeStep(
    inputID: 1, lmHidden: acousticEmbed,
    tailMask: 1.0, textMask: 0,
    kCache: posKIn, vCache: posVIn, positionID: posPosIn)
async let negTask = llm.ttsDecodeStep(
    inputID: 1, lmHidden: acousticEmbed,
    tailMask: 1.0, textMask: 0,
    kCache: negKIn, vCache: negVIn, positionID: negPosIn)
let (posStep, negStep) = try await (posTask, negTask)
```

Per-step diffusion CFG, in `DiffusionRunner.swift`:

```swift
let sampleIn = sample
async let vCondTask = predict(latent: sampleIn, t: reusableTimestep, cond: cond3)
async let vUncondTask = predict(latent: sampleIn, t: reusableTimestep, cond: uncond3)
let (vCond, vUncond) = try await (vCondTask, vUncondTask)
```

### Why the `let` snapshots are necessary

Swift 6 strict concurrency refuses to capture a `var` in `async
let` because "sending a mutable location into a concurrent task
risks a data race". The fix is to bind a `let` copy of each value
immediately before the spawn. `MLMultiArray` is a reference type,
so this doesn't deep-copy the underlying buffer — it just freezes
the Swift-level binding. The CoreML boundary treats the array as
read-only.

### Why the runners had to become `@unchecked Sendable`

`async let` requires the receiver of the method call to be
`Sendable`. `LLMRunner`, `DiffusionRunner`, `VAERunner`,
`HeadsRunner`, and `TTSLoop` are now `final class ... @unchecked
Sendable`. The "unchecked" part claims:

- The stored properties are all CoreML models (`MLModel`), which
  Apple documents as thread-safe for prediction.
- No mutable per-instance scratch state exists. (This was true for
  every runner *except* `LLMRunner`, which previously stored
  decode input MLMultiArrays and mutated them in-place per call.
  That had to be refactored — see next section.)

We also added retroactive `Sendable` conformances on `MLModel` and
`MLMultiArray` in `ModelLoader.swift`:

```swift
extension MLModel: @unchecked @retroactive Sendable {}
extension MLMultiArray: @unchecked @retroactive Sendable {}
```

This is safe given Apple's thread-safety guarantees for prediction
and the fact that we never mutate an array after handing it to
CoreML. If you later add a code path that *does* mutate a shared
`MLMultiArray` concurrently, these annotations will silently lie
and you'll get memory corruption — revisit the conformances.

### The concurrency-safe scratch refactor

`LLMRunner` used to own four stored `MLMultiArray` properties
(`decodeIDs`, `decodePos`, `decodeTail`, `decodeTM`) that it
mutated in-place every call. That was fine sequentially, but would
race when pos and neg `ttsDecodeStep` calls run in parallel — both
calls would clobber the same scratch buffers mid-flight.

Fix: delete the stored scratch, allocate fresh per call via two
small static helpers:

```swift
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
```

The per-call allocation cost (~4 × 1-element MLMultiArrays,
microseconds) is negligible vs the 15–20 ms CoreML forward. Any
future micro-optimization that wants to pool these buffers must
keep the pool per-task, not per-instance.

### Verification

See `scripts/diff_parallel_vs_sequential.py` and
`scripts/locate_divergence.py`. Baseline clips are in
`out/clips/_baseline_*/`; current clips in `out/clips/{a_voices,
b_steps, c_length, d_seeds}/`.

- **20 / 23 clips bit-identical** across the entire waveform
  (max|x−y| = 0.0).
- **3 / 23 clips** differ only in the last 240 samples
  (argmax of |diff| at fraction ≥ 0.998 of the clip). Confirmed
  those are in the raised-cosine fade-out region the earlier
  tail-trim pass produced. Head and middle thirds bit-identical.
  Those 3 clips are also +9600 samples (3 extra speech frames)
  because tail-trim was disabled in a separate change, not from
  anything in the parallelization path.

Parallelization introduces **zero** numerical drift on the
overlapping duration of every clip. If a future regression shows
a non-zero head/middle third max|diff|, that's a real concurrency
bug — investigate.

### RTF impact

| group      | mean Δ  | best Δ   | worst Δ  |
| ---------- | ------: | -------: | -------: |
| all 23     | **−32.2 %** | −53.6 % | −12.1 % |
| voices (6) | −38.1 % | −52.3 % | −29.5 % |
| steps (7)  | −43.9 % | −53.6 % | −30.8 % |
| seeds (7)  | −19.9 % | −22.6 % | −13.5 % |

Step-count-heavy runs benefit most (more CFG pairs per frame).
`steps20_s42` went from 1.21 → 0.66 RTF — i.e. from slower than
real-time to faster. Fastest absolute clip: `b_steps/steps5_s42`
at 0.36 RTF (~2.8× real-time).

---

## Dead end #1: stateful KV cache (`ct.StateType`)

### What I thought

The TTS LM decode graph takes the full K/V cache as
`MLMultiArray` input and returns a new one as output every step.
At T=253 (post-voice-prompt) growing to maybe T=500, the cache
is shape `(L=20, 1, KVH=4, T, D=64)` = up to ~10 MB. Moving that
across the Swift↔CoreML boundary 60 times per clip *felt*
expensive. `ct.StateType` (iOS 18+) lets CoreML own the state
internally; `model.predict(inputs, state: myState)` wouldn't
shuttle the cache at all.

I estimated 20–30 % RTF upside. **Do not trust that estimate.**

### What the measurement actually said

Instrumented `LLMRunner.ttsDecodeStep` with three timers around
the `prediction(from:)` call:

```
  pack  (dict build):     0.019 ms/call   0.1 %
  fwd   (prediction):    17.094 ms/call  99.8 %
  unpack (feature get):   0.011 ms/call   0.1 %
```

Over 56 calls on frank_s42. The Swift-side dict build and
feature retrieval are effectively free. **99.8 % of every step is
inside `model.prediction(from:)`** — GPU forward pass.

Cache-length scaling was the second clue:

| cache T | fwd per call |
| ------: | -----------: |
| 0–49    | 13.5 ms      |
| 250–299 | 19.6 ms      |

+6 ms over a 250-position growth. Some of that is whatever
MLMultiArray↔Metal marshalling CoreML does internally, but most
of it is attention's O(T) cost inside the graph itself — which
stateful can't remove.

Best-case `ct.StateType` upside: maybe 2–5 % RTF, if it even
recovers the internal marshalling. Realistic: **~2 %**. Cost: a
full re-export of `tts_lm_decode.mlpackage` with `register_buffer`
state, per-model parity verification at loose tolerances, plus
`LLMRunner` + `TTSLoop` rewiring to manage two `MLState`
instances (one per CFG branch). Call it 4–8 hours of risk.

Don't do it unless you first re-verify the measurement holds
after any other graph changes. The instrumentation to reproduce
was removed after use, but the pattern is:

```swift
// Add around `model.prediction(from: provider)` in
// LLMRunner.ttsDecodeStep, gated on env var, and call a
// `dumpShuttleProfile()` at end of CLI run.
```

### What `ct.StateType` would actually look like if you tried anyway

You need three things that coremltools makes awkward:

1. **A `register_buffer` for the cache in the PyTorch wrapper.**
   Fixed size `(L, 1, KVH, MAX_T, D)` — you lose the dynamic T
   that the current graph has. Pick MAX_T large enough for the
   longest clip you'll ever synthesize (safe: 2048).

2. **In-place state updates that coremltools traces correctly.**
   The pattern `self.k_cache[i, :, :, P:P+1, :] = k` gets the
   same dynamic-slice complaint the original decode graph hit
   (`ios18.slice_by_index zero-shape bug`). The workaround pattern
   from Apple's WWDC24 KV-cache sample is to use a full read and
   an attention mask derived from position, rather than a slice.

3. **Per-CFG-branch states.** CoreML `MLState` is per-model-
   instance. Either load `tts_lm_decode.mlpackage` twice (doubles
   decode-graph RAM, keeps CFG clean), or bake a batch dim into
   the export and run pos+neg as a single batch=2 forward. The
   second is elegant but conflicts with the current per-branch KV
   cache layout.

If you do all that and the measurement still holds at 99.8 % fwd,
you've spent a day for a ~2 % speedup. Don't.

---

## Dead end #2: post-training fp16 via `linear_quantize_weights`

### The existing fp16 directory

`conversion/out/fp16/` was produced by `09_quantize_fp16.py`,
which uses `coremltools.optimize.coreml.linear_quantize_weights`.
Sizes drop ~4×:

| model              | fp32   | fp16  |
| ------------------ | -----: | ----: |
| tts_lm_decode      | 1.6 GB | 416 MB |
| tts_lm_prefill     | 1.6 GB | (symlink to fp32) |
| text_lm_decode     | 747 MB | 188 MB |
| diffusion_head     | 161 MB | 40 MB |
| vae_decoder        | 1.3 GB | 329 MB |

(Prefills are symlinked to the fp32 versions because fp16
quantization failed on them. Fine — prefill runs once per voice
load.)

### The result

Pointed the CLI at `--models conversion/out/fp16` on frank_s42:

| stage           | fp32   | fp16    | Δ        |
| --------------- | -----: | ------: | -------: |
| text_lm_decode  | 0.10 s | 0.13 s  | +30 %    |
| tts_lm_decode   | 0.66 s | 0.84 s  | +27 %    |
| diffusion       | 0.42 s | 0.55 s  | +31 %    |
| vae_decode      | 0.18 s | **22.08 s** | **+12 000 %** |
| **RTF**         | **0.36** | **7.71** | **~21× slower** |

### Why

`linear_quantize_weights` quantizes *weights* to fp16 and leaves
the *compute graph* at fp32. At predict time each weight is
upcast to fp32 before the op, the op runs fp32, output is fp32.
No GPU fp16 arithmetic win, plus a per-op upcast cost. Worse,
some ops in the VAE don't have the combination of (weight=fp16,
activation=fp32, compute_unit=GPU) available at runtime and fall
back to CPU — hence the 22-second VAE decode.

### Implication

The `fp16/` directory is useful *only* if you care about on-disk
size (e.g. bundling a distribution). It is actively worse for
execution. The CLI has no protection against pointing at it;
maybe add a size check or a manifest note.

### What a real fp16 win would look like

Re-convert each hot-loop model from scratch with

```python
ml_model = ct.convert(
    traced,
    ...,
    compute_precision=ct.precision.FLOAT16,   # <-- here
    minimum_deployment_target=ct.target.iOS18,
)
```

This bakes fp16 into the compute graph itself: activations,
intermediates, and ops all in fp16. Real GPU fp16 arithmetic,
~1.5–2× throughput on M-series.

Per-model risks to verify:

- **text_lm_decode / tts_lm_decode**: Qwen2 SDPA at fp16
  generally works. Parity tolerance loosens from ~1e-3 to ~1e-2
  on hidden states. Check that the post-RMSNorm hidden doesn't
  underflow at long contexts.

- **diffusion_head**: FluidInference's PocketTTS finding was that
  fp16 errors compound across 10 denoising steps, degrading
  quality. Test by ear at multiple step counts (3, 5, 10) and
  compare mel-spec distance vs fp32.

- **vae_decoder**: the 3200× causal-conv upsampler. fp16 overflow
  is a real risk at intermediate activations because the dynamic
  range over the upsampling chain is large. Look at
  `|activation|.max()` at each stage during conversion; if any
  intermediate exceeds ~65 000 you'll get NaNs at predict time.
  Fallback: keep VAE at fp32 (it's only 13 % of per-frame time),
  fp16 everything else.

- **acoustic_connector / tts_eos_classifier**: tiny heads,
  trivial to re-convert. Not in the hot path enough to matter
  but cheap to keep consistent.

Order of attack: tts_lm_decode first (biggest), then
diffusion_head, then text_lm_decode, then VAE (most risky).
Each gated by running `05_parity_check.py` and by a single-clip
listen.

Expected payoff on this hardware: another **25–40 % RTF
reduction** if all five models convert cleanly. If the VAE fails
and you keep it at fp32, probably 15–25 %.

---

## What else is in the plan but not worth today

- **Batched cond+uncond as a single batch=2 forward pass** for
  diffusion. Would remove the async let overhead entirely. But
  that overhead is already small (the two async lets serialize on
  the same GPU anyway; `async let` just lets CoreML's own
  scheduler overlap what it can). Probably 2–5 % upside. Lower
  priority than the fp16 re-export.

- **Apple Neural Engine (ANE) via `ComputeUnit.all`**. The TTS
  LM's KV cache at T=500+ exceeds ANE tensor size limits
  (documented in the original plan). The diffusion head might
  fit — untested. Partial ANE is not obviously a win because
  it fragments the pipeline across compute units and the
  cross-unit handoff cost can dominate.

- **Shorter `max_position` range in exports**. The decode
  graphs were traced with `RangeDim(1, 8192)`. Our actual
  contexts peak around T=1000. Shrinking the range might let
  CoreML's shape inference optimize harder. Unmeasured. Low ROI.

- **Acoustic-connector fusion into the TTS-LM decode graph.**
  Currently a separate `heads.project(latent)` call at 0.01 s
  per frame. Fusing would save the MLMultiArray boundary. But
  the boundary is the same 0.2 % we already measured elsewhere.

---

## Verification methodology

Any future optimization must pass these gates:

### 1. Determinism parity on a fixed-seed clip

Baseline: `out/clips/_baseline_sequential_frank_s42.wav`.
```
./vibevoice-cli "The quick brown fox..." --voice en-Frank_man \
  --steps 5 --seed 42 --output _probe.wav
python3 -c "import soundfile as sf, numpy as np; \
  a, _ = sf.read('baseline.wav'); b, _ = sf.read('_probe.wav'); \
  print(np.abs(a - b).max())"
```
Target: `0.0`. If the change is expected to shift numerics
(e.g. fp16), log the expected magnitude and tighten any
downstream spec-flux tolerance.

### 2. Full-matrix regression

`bash scripts/generate_test_clips.sh` regenerates 23 clips
across 4 axes (voices, steps, length, seeds). Report at
`out/clips/_report.md`.

Diff against the last good baseline via
`python3 scripts/diff_parallel_vs_sequential.py`. The
assertion: 20+/23 bit-identical, non-identical clips must have
differences localized to documented regions (tail-trim fade,
etc.).

### 3. Localize any divergence

`python3 scripts/locate_divergence.py` splits each differing
clip into head/middle/tail thirds and reports max|diff| per
third. Head or middle nonzero = real numeric regression. Only
tail nonzero = fade-boundary issue, annotate and move on.

### 4. RTF comparison

Copy `_report.md` to `_report_<change>.md` before running the
new code; after, the `diff_parallel_vs_sequential.py` output
has an "RTF change" summary row. Per-clip deltas expose any
pathological regression (one clip 3× slower, rest unchanged =
an ANE/GPU dispatch bug).

### 5. Ear check

Open key clips in VLC. Essentials: `a_voices/frank_male.wav`
(bit-identity canary), `a_voices/grace_female.wav` (tail-trim
regression suspect), `c_length/long.wav` (mid-word EOS issue
— known, should not get worse), `b_steps/steps20_s42.wav`
(most stressed pipeline).

---

## Swift 6 concurrency gotchas encountered

Collected for the next time you touch a runner:

- **Retroactive `Sendable` on Apple types.** `MLModel` and
  `MLMultiArray` aren't `Sendable` in the SDK. Marking them
  `@unchecked @retroactive Sendable` in `ModelLoader.swift` is
  the only way to let them cross `async let` boundaries. Safe
  iff (a) the array is read-only after handoff to CoreML,
  (b) the model is used only via `prediction(...)`. Both hold
  in the current code; any new code path that mutates a shared
  array concurrently is a bug these annotations won't catch.

- **`async let` requires `let` capture.** `async let posTask =
  llm.step(kCache: posK)` where `posK` is a `var` fails with
  "sending 'posK' risks causing data races". Bind a `let posKIn
  = posK` immediately before the spawn. The `let` is frozen at
  spawn time, which is what you want.

- **`@unchecked Sendable` on classes.** Required on every runner
  class that gets sent into `async let`. Claims: no mutable
  shared state. If you add a stored property that isn't itself
  `Sendable` and is mutated, the claim is false and you have
  silent races. Per-instance mutable scratch is the usual
  offender (as in the old `LLMRunner`); keep scratch
  per-call.

- **`nonisolated(unsafe)` for shared static mutable state.**
  Used briefly in the removed shuttle profiler. If you need it
  again, lock around every access; Swift 6 will not check for
  you.

---

## File map for the optimization work

```
swift/Sources/VibeVoiceKit/
  TTSLoop.swift           — pos/neg async let, boundary trim
  DiffusionRunner.swift   — cond/uncond async let CFG
  LLMRunner.swift         — per-call scratch allocation
  ModelLoader.swift       — retroactive Sendable conformances
  HeadsRunner.swift       — @unchecked Sendable
  VAERunner.swift         — @unchecked Sendable

scripts/
  generate_test_clips.sh       — 23-clip regression matrix
  diff_parallel_vs_sequential.py — RTF + quality diff
  locate_divergence.py         — localize non-bit-identical clips

out/clips/
  _baseline_a_voices/          — pre-parallel WAVs, set A
  _baseline_b_steps/           — pre-parallel WAVs, set B
  _baseline_c_length/          — pre-parallel WAVs, set C
  _baseline_d_seeds/           — pre-parallel WAVs, set D
  _report_baseline_sequential.md — pre-parallel RTF report
  _report.md                   — current RTF report
  _parallel_vs_sequential.md   — diff output

docs/
  conversion-notes.md     — chronological log, append-only
  performance-optimization.md — this file
```

---

## If you're here to make it faster

Read, in order:

1. The TL;DR.
2. "What a real fp16 win would look like" under Dead End #2.
3. The verification methodology.

Then re-run the shuttle profiler (add it back; pattern is in
the parallelization section) on *your* hardware before
committing to anything big. What's true on an M2 may be
different on an M4 or iPhone. The 0.2 % shuttle number is the
single most load-bearing assumption in this document; if it's
different on your device, the stateful rewrite might suddenly
be worthwhile after all.
