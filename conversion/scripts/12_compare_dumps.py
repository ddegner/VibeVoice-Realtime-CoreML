"""
Compare PyTorch reference (10_pytorch_reference_dump.py) vs CoreML trace
(11_coreml_trace_dump.py) and report max|Δ| at every stage. The first stage
whose diff exceeds the threshold is the culprit.

Usage:
  # 1. Dump PyTorch reference (the ground truth)
  python scripts/10_pytorch_reference_dump.py \
      --text "..." --voice en-Frank_man --seed 42 --steps 5 --frames 3 \
      --tag fox_s42_st5

  # 2. Dump CoreML-via-Python trace (uses PyTorch's noise for parity)
  python scripts/11_coreml_trace_dump.py \
      --text "..." --voice en-Frank_man --seed 42 --steps 5 --frames 3 \
      --tag fox_s42_st5 --use-ref-noise

  # 3. Compare
  python scripts/12_compare_dumps.py --tag fox_s42_st5

Interpretation:
  * text_tokens diff  → tokenizer mismatch
  * tts_lm_first_window diff  → TTS LM conversion or text-window feeding bug
  * posLast_frame{f} diff  → acoustic feedback drift (either connector, or
      TTS-LM decode on speech-frame positions)
  * noise_frame{f} diff  → RNG mismatch (expected unless --use-ref-noise)
  * vcond/vuncond_frame{f}_step{s} diff  → diffusion head conversion bug
  * v_frame{f}_step{s} diff  → CFG blending mismatch (should match iff v_cond
      and v_uncond match)
  * x0_frame{f}_step{s} diff  → scheduler alpha/sigma mismatch
  * sample_frame{f}_step{s} diff  → scheduler update formula mismatch
  * latent_frame{f} diff  → cumulative across all 5-10 steps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402


def compare(a: np.ndarray, b: np.ndarray, label: str, tol: float = 1e-3) -> tuple[float, bool]:
    if a.shape != b.shape:
        print(f"  {label}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return (float("inf"), True)
    diff = float(np.max(np.abs(a - b)))
    rel = diff / max(float(np.max(np.abs(a))), 1e-12)
    status = "" if diff < tol else "  ⚠️ FAIL"
    print(f"  {label:40s}  max|Δ|={diff:10.3e}  rel={rel:8.1e}{status}")
    return (diff, diff >= tol)


def load_pair(ref_dir: Path, coreml_dir: Path, name: str):
    a = ref_dir / name
    b = coreml_dir / name
    if not (a.exists() and b.exists()):
        return None, None
    return np.load(a), np.load(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    ref = OUT_DIR / "ref" / args.tag
    cm = ref / "coreml"
    if not ref.exists():
        raise SystemExit(f"{ref} not found. Run 10_pytorch_reference_dump.py first.")
    if not cm.exists():
        raise SystemExit(f"{cm} not found. Run 11_coreml_trace_dump.py first.")

    ref_meta = json.loads((ref / "meta.json").read_text())
    cm_meta = json.loads((cm / "meta.json").read_text())
    frames = min(ref_meta["frames_dumped"], cm_meta["frames_dumped"])
    steps = ref_meta["steps"]
    print(f"Comparing {args.tag} ({frames} frames × {steps} steps, tol={args.tol})")
    print()

    any_fail = False

    # 1. Tokens
    a, b = load_pair(ref, cm, "text_tokens.npy")
    if a is not None:
        if a.shape == b.shape and np.array_equal(a, b):
            print("  text_tokens                              match")
        else:
            print(f"  text_tokens                              MISMATCH: {a[:10]} vs {b[:10]}")
            any_fail = True

    # 2. TTS-LM first text-window hidden
    a, b = load_pair(ref, cm, "tts_lm_first_window.npy")
    if a is not None:
        _, fail = compare(a, b, "tts_lm_first_window (per-token posLast)", tol=args.tol)
        any_fail |= fail

    print()

    # 3. Per-frame intermediates
    for f in range(frames):
        print(f"Frame {f}:")
        for name, tol_override in [
            (f"posLast_frame{f}",   args.tol),
            (f"negLast_frame{f}",   args.tol),
            (f"noise_frame{f}",     1e-5),  # should be bit-exact if using ref noise
        ]:
            a, b = load_pair(ref, cm, f"{name}.npy")
            if a is not None:
                _, fail = compare(a, b, name, tol=tol_override)
                any_fail |= fail

        for s in range(steps):
            for name in [
                f"vcond_frame{f}_step{s}",
                f"vuncond_frame{f}_step{s}",
                f"v_frame{f}_step{s}",
                f"sample_frame{f}_step{s}",
            ]:
                a, b = load_pair(ref, cm, f"{name}.npy")
                if a is not None:
                    _, fail = compare(a, b, name, tol=args.tol)
                    any_fail |= fail

        a, b = load_pair(ref, cm, f"latent_frame{f}.npy")
        if a is not None:
            _, fail = compare(a, b, f"latent_frame{f} (final)", tol=args.tol)
            any_fail |= fail
        print()

    print("=" * 60)
    print("FAIL" if any_fail else "PASS")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
