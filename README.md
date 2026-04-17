# VibeVoice-Realtime for CoreML

A local CoreML port of [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B), with a Swift library, a CLI, and a SwiftUI shell. Targets macOS 15+ on Apple Silicon.

End-to-end synthesis runs at roughly **0.4× real-time** on an M-series Mac (5 diffusion steps, CFG 1.5).

## Status

Working: voice prompt loading, text → speech synthesis, 24 voices, deterministic seeding, sliding-window VAE decode, CFG, fp16 quantized weights. Streaming (Phase G in the plan) is not yet wired.

The repository ships **code only** — no model weights or `.mlpackage` artifacts. Run the conversion scripts against the HuggingFace checkpoint to produce them locally.

## Layout

- `conversion/` — Python scripts that convert the PyTorch checkpoint into the CoreML artifacts (text LM prefill+decode, TTS LM prefill+decode, diffusion head, VAE decoder, acoustic connector, EOS classifier).
- `swift/` — `VibeVoiceKit` library, `vibevoice-cli` test harness, and `VibeVoiceApp` SwiftUI shell.
- `docs/` — running notes on conversion pitfalls and performance.
- `scripts/` — diagnostic helpers (PyTorch ↔ CoreML divergence locator, etc.).

## Quick start (conversion)

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Vendor the upstream repo (gitignored)
git clone https://github.com/microsoft/VibeVoice third_party/VibeVoice
pip install -e third_party/VibeVoice

python scripts/01_inspect_model.py        # writes out/model_manifest.json
python scripts/02_convert_llm.py          # text LM prefill + decode
python scripts/02b_convert_tts_lm.py      # TTS LM prefill + decode
python scripts/03_convert_diffusion.py    # diffusion head
python scripts/04_convert_vae.py          # VAE decoder
python scripts/05_bundle_tokenizer.py     # tokenizer assets
python scripts/06_convert_heads.py        # acoustic_connector + tts_eos_classifier
python scripts/08_export_voice_prompt.py  # per-voice KV caches (.npy)
python scripts/09_quantize_fp16.py        # optional: fp16 weights, ~2× smaller
```

Artifacts land in `conversion/out/` (gitignored).

## Quick start (Swift)

```bash
cd swift
swift build -c release
swift run -c release vibevoice-cli "Hello from CoreML." \
    --models ../conversion/out \
    --voice en-Frank_man \
    -o /tmp/out.wav
```

Useful flags: `--steps`, `--guidance`, `--seed`, `--list-voices`, `--play`, `--trace-eos`.

## Architecture

Eight CoreML models orchestrated by Swift (CoreML cannot express the iterative DDPM loop natively):

1. **Text LM prefill / decode** — Qwen2.5-0.5B backbone, stateful KV cache.
2. **TTS LM prefill / decode** — second LM (434M params) conditioned on text LM hidden states, with tail-splice + type-embedding for text/speech token positions.
3. **Diffusion head** — single-step v-prediction denoiser, called per frame with CFG (DPM-Solver++ order 2 in Swift).
4. **VAE decoder** — acoustic latents → 24 kHz waveform (3200× upsampling).
5. **Acoustic connector** — projects (1, T, 64) latents into (1, T, 896) LM hidden space for autoregressive feedback.
6. **TTS EOS classifier** — sigmoid head over the last TTS LM hidden state.

See [`docs/conversion-notes.md`](docs/conversion-notes.md) for the running log of conversion pitfalls and [`docs/performance-optimization.md`](docs/performance-optimization.md) for tuning notes.

## License

This repository is MIT-licensed — see [`LICENSE`](LICENSE).

The upstream [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) code is also MIT-licensed (Copyright Microsoft 2025). The model weights are governed by their own terms on HuggingFace; check the model card before redistributing any derived `.mlpackage` artifacts.

## Acknowledgments

- Microsoft Research for the VibeVoice model and reference implementation.
- The dots.ocr blog and FluidInference's mobius work, both cited in [`docs/conversion-notes.md`](docs/conversion-notes.md), for prior art on Qwen2.5 → CoreML and diffusion-head conversion patterns.
