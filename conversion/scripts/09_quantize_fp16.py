"""
Post-training fp16 weight quantization of CoreML models.

Uses coremltools 9.0's `linear_quantize_weights` to convert fp32 weights to
fp16, halving model size and improving GPU throughput on Apple Silicon.

Usage:
  python scripts/09_quantize_fp16.py
  python scripts/09_quantize_fp16.py --only tts_lm_decode
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402

FP16_DIR = OUT_DIR / "fp16"

MODELS = [
    "tts_lm_decode",
    "tts_lm_prefill",
    "text_lm_decode",
    "text_lm_prefill",
    "diffusion_head",
    "vae_decoder",
    "acoustic_connector",
    "tts_eos_classifier",
]


def quantize_model(name: str) -> None:
    src = OUT_DIR / f"{name}.mlpackage"
    dst = FP16_DIR / f"{name}.mlpackage"
    if not src.exists():
        print(f"  SKIP {name} (not found)")
        return
    if dst.exists():
        print(f"  SKIP {name} (already exists at {dst})")
        return

    print(f"  Loading {name}…")
    model = ct.models.MLModel(str(src))

    print(f"  Quantizing weights to fp16…")
    op_config = OpLinearQuantizerConfig(dtype="float16")
    config = OptimizationConfig(global_config=op_config)
    model_fp16 = linear_quantize_weights(model, config=config)

    print(f"  Saving to {dst}…")
    model_fp16.save(str(dst))

    src_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"  {name}: {src_mb:.0f} MB → {dst_mb:.0f} MB ({dst_mb/src_mb:.1%})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, help="Only quantize this model")
    args = parser.parse_args()

    FP16_DIR.mkdir(parents=True, exist_ok=True)

    # Link/copy non-model assets
    for item in ["voices", "tokenizer", "speech_constants.json"]:
        src = OUT_DIR / item
        dst = FP16_DIR / item
        if src.exists() and not dst.exists():
            if src.is_dir():
                print(f"  Linking {item}/")
                dst.symlink_to(src.resolve())
            else:
                print(f"  Copying {item}")
                shutil.copy2(src, dst)

    models = [args.only] if args.only else MODELS
    for name in models:
        print(f"\n[{name}]")
        quantize_model(name)

    print("\nDone! fp16 models in:", FP16_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
