"""Shared helpers for the conversion scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("VIBEVOICE_OUT_DIR", ROOT / "out")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = Path(
    os.environ.get("VIBEVOICE_MANIFEST_PATH", OUT_DIR / "model_manifest.json")
).expanduser()


def load_manifest() -> dict[str, Any]:
    candidates = [MANIFEST_PATH]
    default_manifest = ROOT / "out" / "model_manifest.json"
    if default_manifest != MANIFEST_PATH:
        candidates.append(default_manifest)

    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())

    searched = ", ".join(str(path) for path in candidates)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"model_manifest.json not found at {searched} — run scripts/01_inspect_model.py first."
        )


def coreml_compute_precision(ct):
    raw = coreml_compute_precision_name()
    if raw in {"float32", "fp32"}:
        return ct.precision.FLOAT32
    if raw in {"float16", "fp16"}:
        return ct.precision.FLOAT16
    raise ValueError(
        "VIBEVOICE_COMPUTE_PRECISION must be float32/fp32 or float16/fp16, "
        f"got {raw!r}"
    )


def coreml_compute_precision_name() -> str:
    raw = os.environ.get("VIBEVOICE_COMPUTE_PRECISION", "float32").lower()
    return raw.replace("-", "").replace("_", "")


def parity_tolerance(float32: float, float16: float) -> float:
    return float16 if coreml_compute_precision_name() in {"float16", "fp16"} else float32


def resolve(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def load_vibevoice(dtype=torch.float32):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )

    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=dtype,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    return model


def assert_parity(a: torch.Tensor, b, tol: float, label: str) -> None:
    """Compare a PyTorch tensor vs either another tensor or a numpy array."""
    import numpy as np

    a_np = a.detach().float().cpu().numpy()
    b_np = b if isinstance(b, np.ndarray) else b.detach().float().cpu().numpy()
    diff = float(np.max(np.abs(a_np - b_np)))
    print(f"[parity] {label}: max|Δ| = {diff:.3e} (tol {tol:.1e})")
    if diff > tol:
        raise SystemExit(f"[parity] FAIL {label}: {diff:.3e} > {tol:.1e}")
