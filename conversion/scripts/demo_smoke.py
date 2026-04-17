"""
Smoke-test the converted .mlpackages end-to-end *as CoreML graphs*.

This is NOT real synthesis — no real text, no scheduler, no trained conditioning.
It's just "do the models load and produce output of the expected shape?"

  1. Load all 6 .mlpackages.
  2. Run VAE with random latents → WAV (will be noise, proves VAE works).
  3. Run diffusion head once with random inputs → shape check.
  4. Run text LM prefill with the BOS token → hidden-state shape check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parents[1] / "out"


def load(name: str) -> ct.models.MLModel:
    print(f"  loading {name}.mlpackage…")
    return ct.models.MLModel(str(OUT / f"{name}.mlpackage"))


def main() -> int:
    print("=== Loading all CoreML models ===")
    vae = load("vae_decoder")
    diff = load("diffusion_head")
    text_prefill = load("text_lm_prefill")
    tts_prefill = load("tts_lm_prefill")

    print("\n=== [1] VAE: random latents → 24 kHz audio ===")
    # (1, 64, 16) — fixed T=16 as traced. Scale roughly to trained latent range.
    lat = np.random.randn(1, 64, 16).astype(np.float32) * 0.5
    vae_out = vae.predict({"latent": lat})
    wav = vae_out["waveform"].reshape(-1)
    print(f"  output: {wav.shape}, range [{wav.min():.6f}, {wav.max():.6f}], rms={np.sqrt((wav**2).mean()):.6f}")
    print(f"  duration: {len(wav) / 24000:.3f} s")
    out_path = OUT / "demo_noise.wav"
    # Clip and write 16-bit PCM so it's universally playable.
    wav_clip = np.clip(wav, -1.0, 1.0)
    sf.write(out_path, wav_clip, 24000, subtype="PCM_16")
    print(f"  wrote {out_path}  (open it — should be noise)")

    print("\n=== [2] Diffusion head: one forward pass ===")
    noisy = np.random.randn(1, 8, 64).astype(np.float32)
    t = np.array([500.0], dtype=np.float32)
    cond = np.random.randn(1, 8, 896).astype(np.float32) * 0.1
    pred = diff.predict({"noisy_latent": noisy, "timestep": t, "conditioning": cond})
    eps = pred["predicted_noise"]
    print(f"  predicted_noise shape {eps.shape}  range [{eps.min():.3f}, {eps.max():.3f}]")

    print("\n=== [3] Text LM prefill with BOS=151643 (T=16, trace-baked) ===")
    # Trace baked T=16 despite RangeDim; pad with BOS to satisfy shape.
    ids = np.full((1, 16), 151643, dtype=np.int32)
    tp_out = text_prefill.predict({"input_ids": ids})
    h = tp_out["hidden_states"]
    k = tp_out["k_cache"]
    v = tp_out["v_cache"]
    print(f"  hidden_states {h.shape}  k_cache {k.shape}  v_cache {v.shape}")

    print("\n=== [4] TTS LM prefill (mask form) ===")
    lm_h = np.random.randn(1, 16, 896).astype(np.float32) * 0.1
    tail = np.ones((1, 16, 1), dtype=np.float32)    # full override
    tm = np.ones((1, 16), dtype=np.int32)            # all-text positions
    ttp_out = tts_prefill.predict({
        "input_ids": ids,
        "lm_hidden_padded": lm_h,
        "tail_mask": tail,
        "tts_text_masks": tm,
    })
    print(f"  hidden_states {ttp_out['hidden_states'].shape}  k_cache {ttp_out['k_cache'].shape}")

    print("\n=== [5] Heads: acoustic_connector + tts_eos_classifier ===")
    conn = load("acoustic_connector")
    eos = load("tts_eos_classifier")
    lat1 = np.random.randn(1, 1, 64).astype(np.float32) * 0.5
    emb = conn.predict({"latent": lat1})["embedding"]
    print(f"  connector out {emb.shape}")
    hid = np.random.randn(1, 896).astype(np.float32) * 0.1
    p = eos.predict({"hidden": hid})["prob"]
    print(f"  eos prob {p.shape} = {p.flatten()[0]:.4f}")

    print("\nAll models executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
