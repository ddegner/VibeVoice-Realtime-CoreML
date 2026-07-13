# VibeVoice speed hypothesis results - 2026-07-09

Hardware: M1 Max, 64 GB RAM, macOS 26.5.1. All synthesis tests used the
shipping INT8-weight/FP32-compute models unless a row explicitly says
otherwise, Carter voice, seed 42, guidance 1.5, and five diffusion steps.
The fixture generated 24.67 seconds of baseline audio from 412 characters.

## Results

| Hypothesis | Measured result | Audio result | Decision |
| --- | --- | --- | --- |
| Shared-engine two-chunk concurrency | 21.83s sequential versus 15.88s concurrent; 27.3% less wall time and 37.5% more aggregate throughput | All outputs byte-identical to baseline; peak RSS 8.42 GB versus 8.53 GB sequential | Implement with concurrency limited to two |
| Parallel model loading | Mean model load 49.06s sequential versus 43.08s parallel; 12.2% faster | Byte-identical | Implement; modest startup improvement |
| Batch-2 CFG models | 16.70s versus 11.42s baseline; 46.2% slower. TTS-LM 7.83s versus 5.88s; diffusion 6.62s versus 3.29s | Same duration, 56.15 dB SNR, 0.9999988 correlation | Reject |
| Five-token block decode | 10.34s; 5.3% faster than the two-run sequential average and 9.5% faster than the nearest single baseline | Same duration, but 0.559 correlation and glitch proxy 259 versus 216 | Do not ship without a favorable listening test; gain is modest |
| True FP16 TTS-LM compute | Reached the 500-frame cap in 24.18s instead of stopping at 191 frames | Invalid output: 66.67s, peak amplitude 1,629,330, broken EOS/acoustic feedback | Reject |

Baseline single-run synthesis varied from 10.92s per clip in the two-run
measurement to 12.26s in the initial cold process. Percentage claims above
use the closest comparable run and are not extrapolated from one isolated
number.

## Listening files

- `listening/20260709-hypotheses/baseline.wav`
- `listening/20260709-hypotheses/batched-cfg.wav`
- `listening/20260709-hypotheses/block5.wav`

The FP16 output is intentionally omitted from the listening set because its
amplitude is unsafe and the generation is objectively invalid.

## Quality checks

- Per-synthesis diffusion state produced a byte-for-byte identical baseline
  before concurrency was measured.
- Concurrent outputs, sequential-repeat outputs, and the parallel-load output
  all have the same SHA-256 as the original baseline.
- Batch-2 CFG preserved the waveform closely but was slower in both hot stages.
- Block decode preserved duration and broad amplitude statistics, but changed
  the autoregressive trajectory enough that waveform comparison cannot certify
  subjective quality.

## Recommendation

Move per-synthesis diffusion state and a two-worker chunk scheduler into the
app. Preserve output ordering and use the current five-step model. Also load
independent CoreML artifacts concurrently. Do not ship batch-2 CFG, true FP16
TTS-LM, lower diffusion steps, or VAE-on-ANE.

## Validation

- `swift test` in the CoreML Swift package: 1 test passed.
- `swift test --package-path OratorKit`: 324 tests in 26 suites passed; the
  opt-in end-to-end generation test was skipped.
- `git diff --check` passed in the CoreML package.

The app currently serializes synthesis through its existing gate. These tests
exercise prototype package and CLI paths; no app build was installed as part of
this benchmark run.
