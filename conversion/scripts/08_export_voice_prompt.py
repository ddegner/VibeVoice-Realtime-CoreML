"""
Phase E — Export voice prompt caches as numpy arrays for Swift.

The upstream `.pt` voice files contain pre-computed KV caches for all four
branches (lm, tts_lm, neg_lm, neg_tts_lm). These encode the voice identity
and are required for intelligible speech. Without them the TTS LM produces
noise.

Each branch has:
  - last_hidden_state: (1, T, 896)
  - past_key_values: DynamicCache with L layers of (1, KVH, T, D)

We stack them into the (L, 1, KVH, T, D) tensor format our CoreML decode
models expect and save as .npy files.

Emits (per voice, e.g. "en-Frank_man"):
  out/voices/en-Frank_man/
    lm_hidden.npy          (1, 72, 896) float32
    lm_k_cache.npy         (4, 1, 2, 72, 64) float32
    lm_v_cache.npy         (4, 1, 2, 72, 64) float32
    tts_lm_hidden.npy      (1, 253, 896) float32
    tts_lm_k_cache.npy     (20, 1, 2, 253, 64) float32
    tts_lm_v_cache.npy     (20, 1, 2, 253, 64) float32
    neg_lm_hidden.npy      (1, 1, 896) float32
    neg_lm_k_cache.npy     (4, 1, 2, 1, 64) float32
    neg_lm_v_cache.npy     (4, 1, 2, 1, 64) float32
    neg_tts_lm_hidden.npy  (20, 1, 2, 1, 64) float32
    neg_tts_lm_k_cache.npy
    neg_tts_lm_v_cache.npy
    metadata.json          {lm_len, tts_lm_len, neg_lm_len, neg_tts_lm_len}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402

VOICES_DIR = Path(__file__).resolve().parents[1] / "third_party" / "VibeVoice" / "demo" / "voices" / "streaming_model"


def export_branch(cp_branch: dict, prefix: str, out_dir: Path) -> int:
    """Export one branch (lm, tts_lm, neg_lm, neg_tts_lm) to .npy files.
    Returns the T (sequence length) of the cache."""
    h = cp_branch["last_hidden_state"].float().cpu().numpy()
    kv = cp_branch["past_key_values"]
    ks = [k.float().cpu() for k in kv.key_cache]
    vs = [v.float().cpu() for v in kv.value_cache]
    # Stack (L, 1, KVH, T, D)
    k_stack = torch.stack(ks, dim=0).numpy()
    v_stack = torch.stack(vs, dim=0).numpy()

    np.save(str(out_dir / f"{prefix}_hidden.npy"), h)
    np.save(str(out_dir / f"{prefix}_k_cache.npy"), k_stack)
    np.save(str(out_dir / f"{prefix}_v_cache.npy"), v_stack)
    print(f"  {prefix}: hidden {h.shape}, k {k_stack.shape}, v {v_stack.shape}")
    return h.shape[1]


def main() -> int:
    voices = sorted(VOICES_DIR.glob("*.pt"))
    if not voices:
        print(f"No .pt files found in {VOICES_DIR}")
        return 1

    for vpt in voices:
        name = vpt.stem
        out_dir = OUT_DIR / "voices" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} ===")
        cp = torch.load(str(vpt), map_location="cpu", weights_only=False)

        meta = {}
        for branch in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
            T = export_branch(cp[branch], branch, out_dir)
            meta[f"{branch}_len"] = T

        with (out_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"  metadata: {meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
