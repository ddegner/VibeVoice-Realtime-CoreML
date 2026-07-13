#!/usr/bin/env python3
"""Prepare a side-by-side CoreML compute-precision variant.

This does not touch the app's model directory. It creates a new model directory
that symlinks the baseline artifacts, then re-runs selected conversion scripts
with VIBEVOICE_COMPUTE_PRECISION set to float16 or float32.

Examples:
  python scripts/13_prepare_precision_variant.py --models tts_lm_decode
  python scripts/13_prepare_precision_variant.py --models tts_lm_decode,diffusion_head,text_lm_decode
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_BASE = ROOT / "out" / "fp16"
DEFAULT_OUT = ROOT / "out" / "float16-compute"
DEFAULT_MANIFEST = ROOT / "out" / "model_manifest.json"

MODEL_GROUPS = {
    "text_lm_decode": {
        "outputs": ["text_lm_prefill", "text_lm_decode"],
        "command": ["02_convert_llm.py"],
    },
    "tts_lm_decode": {
        "outputs": ["tts_lm_prefill", "tts_lm_decode"],
        "command": ["02b_convert_tts_lm.py"],
    },
    "diffusion_head": {
        "outputs": ["diffusion_head"],
        "command": ["03_convert_diffusion.py"],
    },
    "vae_decoder": {
        "outputs": ["vae_decoder"],
        "command": ["04_convert_vae.py"],
    },
    "heads": {
        "outputs": ["acoustic_connector", "tts_eos_classifier"],
        "command": ["06_convert_heads.py"],
    },
}


def parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(models).difference(MODEL_GROUPS))
    if unknown:
        known = ", ".join(MODEL_GROUPS)
        raise SystemExit(f"Unknown model group(s): {', '.join(unknown)}. Known: {known}")
    return models


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def symlink_baseline(base: Path, out: Path, force: bool) -> None:
    if not base.exists():
        raise SystemExit(f"Base model directory not found: {base}")
    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        target = out / item.name
        if target.exists() or target.is_symlink():
            if force:
                remove_path(target)
            else:
                continue
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def remove_group_outputs(out: Path, groups: list[str]) -> None:
    for group in groups:
        for stem in MODEL_GROUPS[group]["outputs"]:
            for suffix in (".mlpackage", ".mlmodelc"):
                path = out / f"{stem}{suffix}"
                if path.exists() or path.is_symlink():
                    remove_path(path)


def ensure_manifest() -> Path:
    if DEFAULT_MANIFEST.exists():
        return DEFAULT_MANIFEST
    subprocess.run([sys.executable, str(SCRIPTS / "01_inspect_model.py")], cwd=ROOT, check=True)
    if not DEFAULT_MANIFEST.exists():
        raise SystemExit(f"Manifest still missing after inspection: {DEFAULT_MANIFEST}")
    return DEFAULT_MANIFEST


def run_conversion(out: Path, manifest: Path, precision: str, groups: list[str]) -> None:
    env = os.environ.copy()
    env["VIBEVOICE_OUT_DIR"] = str(out)
    env["VIBEVOICE_MANIFEST_PATH"] = str(manifest)
    env["VIBEVOICE_COMPUTE_PRECISION"] = precision

    for group in groups:
        command = [sys.executable, str(SCRIPTS / MODEL_GROUPS[group]["command"][0])]
        print(f"\n[{group}] {' '.join(command)}")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-models",
        type=Path,
        default=DEFAULT_BASE,
        help="Existing model directory to symlink for unchanged artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Variant model directory to create.",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "fp16", "float32", "fp32"],
        default="float16",
        help="Core ML compute_precision used during conversion.",
    )
    parser.add_argument(
        "--models",
        default="tts_lm_decode",
        help="Comma-separated model groups: text_lm_decode,tts_lm_decode,diffusion_head,vae_decoder,heads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks/artifacts in the variant directory.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the symlinked variant directory; do not run conversion.",
    )
    args = parser.parse_args()

    base = args.base_models.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    groups = parse_models(args.models)

    symlink_baseline(base, out, force=args.force)

    if not args.prepare_only:
        remove_group_outputs(out, groups)
        manifest = ensure_manifest()
        run_conversion(out, manifest, args.precision, groups)

    print(f"\nVariant ready: {out}")
    print("Benchmark it from swift/ with:")
    print(f"  FP16_MODELS_DIR={out} bench/run_speed_quality_matrix.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
