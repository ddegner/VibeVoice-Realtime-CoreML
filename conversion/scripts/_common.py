"""Shared helpers for the conversion scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
MANIFEST_PATH = OUT_DIR / "model_manifest.json"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found — run scripts/01_inspect_model.py first."
        )
    return json.loads(MANIFEST_PATH.read_text())


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
