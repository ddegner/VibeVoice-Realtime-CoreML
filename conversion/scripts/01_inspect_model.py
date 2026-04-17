"""
Phase A.3 — Load VibeVoice-Realtime-0.5B, walk its module tree, and emit a
JSON manifest that every downstream conversion script reads.

Writes: conversion/out/model_manifest.json

Nothing here is CoreML-specific: this is pure introspection. Run this first;
if the attribute paths drift in a future VibeVoice release, this is the only
file that needs updating.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

OUT_DIR = Path(__file__).resolve().parents[1] / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "model_manifest.json"

MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"


def _first_existing(obj, paths):
    for p in paths:
        cur = obj
        ok = True
        for part in p.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return p
    return None


def main() -> int:
    # Import lazily so `--help` etc. don't pay the import cost.
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )

    print(f"Loading {MODEL_ID} on CPU (float32)…", flush=True)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()

    # Candidate attribute paths — based on reading
    # vibevoice/modular/modeling_vibevoice_streaming_inference.py @ 17f57f8.
    # If any return None, inspect the module tree dump below and update.
    paths = {
        "text_lm":         _first_existing(model, ["model.language_model"]),
        "tts_lm":          _first_existing(model, ["model.tts_language_model"]),
        "diffusion_head":  _first_existing(model, ["model.prediction_head"]),
        "acoustic_tok":    _first_existing(model, ["model.acoustic_tokenizer"]),
        "acoustic_decoder":_first_existing(model, [
            "model.acoustic_tokenizer.decoder",
            "model.acoustic_tokenizer.vae.decoder",
        ]),
        "noise_scheduler": _first_existing(model, ["model.noise_scheduler"]),
        "lm_head":         _first_existing(model, ["lm_head"]),
        "tts_eos":         _first_existing(model, ["tts_eos_classifier"]),
    }

    cfg = model.config
    decoder_cfg = getattr(cfg, "decoder_config", None)
    diff_cfg = getattr(cfg, "diffusion_head_config", None)
    ac_cfg = getattr(cfg, "acoustic_tokenizer_config", None) or getattr(
        cfg, "acoustic_vae_dim", None
    )

    manifest = {
        "model_id": MODEL_ID,
        "paths": paths,
        "config": {
            "hidden_size":           getattr(decoder_cfg, "hidden_size", None),
            "num_attention_heads":   getattr(decoder_cfg, "num_attention_heads", None),
            "num_key_value_heads":   getattr(decoder_cfg, "num_key_value_heads", None),
            "num_hidden_layers":     getattr(decoder_cfg, "num_hidden_layers", None),
            "head_dim":              (
                getattr(decoder_cfg, "hidden_size", 0)
                // max(getattr(decoder_cfg, "num_attention_heads", 1), 1)
            ) or None,
            "vocab_size":            getattr(decoder_cfg, "vocab_size", None),
            "max_position":          getattr(decoder_cfg, "max_position_embeddings", None),
            "rope_theta":            getattr(decoder_cfg, "rope_theta", None),
            "ddpm_num_inference_steps": getattr(diff_cfg, "ddpm_num_inference_steps", None),
            "acoustic_latent_dim":   ac_cfg if isinstance(ac_cfg, int) else getattr(ac_cfg, "latent_dim", None),
            "sample_rate":           24000,  # documented for VibeVoice
        },
        "ddpm_inference_steps_runtime": getattr(model, "ddpm_inference_steps", None),
        "param_counts": {
            name: sum(p.numel() for p in mod.parameters())
            for name, attr in paths.items()
            if attr is not None
            for mod in [_resolve(model, attr)]
            if isinstance(mod, torch.nn.Module)
        },
    }

    # Dump the full module tree next to the manifest for reference.
    tree_path = OUT_DIR / "module_tree.txt"
    with tree_path.open("w") as f:
        for name, mod in model.named_modules():
            f.write(f"{name}\t{type(mod).__name__}\n")

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest → {MANIFEST_PATH}")
    print(f"Module tree → {tree_path}")
    print(json.dumps(manifest, indent=2))

    missing = [k for k, v in paths.items() if v is None]
    if missing:
        print(
            f"\n[warn] Could not resolve attribute paths for: {missing}. "
            "Inspect module_tree.txt and update the candidate list in this script.",
            file=sys.stderr,
        )
        return 1
    return 0


def _resolve(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


if __name__ == "__main__":
    sys.exit(main())
