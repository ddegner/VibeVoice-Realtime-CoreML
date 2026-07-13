# Speed and Quality Experiments

Use this bench folder to compare speed and listen to generated WAVs before
changing Orator's app runtime.

## Compute-Unit Routing

From `VibeVoice-Realtime for CoreML/swift`:

```bash
bench/run_speed_quality_matrix.sh
```

The script writes a dated folder under `bench/listening/` with:

- `results.jsonl`: raw per-run timings and audio stats.
- `summary.tsv`: one row per experiment label.
- `*.wav`: generated clips for listening.

Useful overrides:

```bash
RUNS=3 VOICE=en-Carter_man bench/run_speed_quality_matrix.sh
INCLUDE_DIFFUSION_ANE=1 bench/run_speed_quality_matrix.sh
```

## True FP16 Compute

First prepare a side-by-side model directory. This re-exports selected Core ML
models with `compute_precision=FLOAT16`; it is not the same as weight-only
`linear_quantize_weights`.

From `VibeVoice-Realtime for CoreML/conversion`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/13_prepare_precision_variant.py --models tts_lm_decode
```

Then benchmark it from `VibeVoice-Realtime for CoreML/swift`:

```bash
FP16_MODELS_DIR=../conversion/out/float16-compute bench/run_speed_quality_matrix.sh
```

For a broader experiment:

```bash
python scripts/13_prepare_precision_variant.py \
  --force \
  --models tts_lm_decode,diffusion_head,text_lm_decode,heads
```

Keep VAE as a separate experiment; it has the highest overflow and quality risk.
