#!/usr/bin/env python3
"""Export fixed five-token text-window decode models for benchmarking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, load_manifest, load_vibevoice, resolve  # noqa: E402


def decode_layers(lm, x, k_cache, v_cache, position_ids, attention_mask):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    cos, sin = lm.rotary_emb(x, position_ids)
    new_ks, new_vs = [], []
    for index, layer in enumerate(lm.layers):
        residual = x
        h = layer.input_layernorm(x)
        attn = layer.self_attn
        q = attn.q_proj(h)
        k = attn.k_proj(h)
        v = attn.v_proj(h)
        batch, query_len, _ = q.shape
        query_heads = attn.config.num_attention_heads
        kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        q = q.view(batch, query_len, query_heads, head_dim).transpose(1, 2)
        k = k.view(batch, query_len, kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, query_len, kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k_full = torch.cat([k_cache[index], k], dim=2)
        v_full = torch.cat([v_cache[index], v], dim=2)
        new_ks.append(k_full)
        new_vs.append(v_full)
        repeats = query_heads // kv_heads
        k_expanded = k_full.repeat_interleave(repeats, dim=1)
        v_expanded = v_full.repeat_interleave(repeats, dim=1)
        attention = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=attention_mask,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).reshape(
            batch, query_len, query_heads * head_dim)
        x = residual + attn.o_proj(attention)
        residual = x
        x = residual + layer.mlp(layer.post_attention_layernorm(x))
    return lm.norm(x), torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


class TextBlockWrapper(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, k_cache, v_cache, position_ids, attention_mask):
        x = self.lm.embed_tokens(input_ids)
        return decode_layers(
            self.lm, x, k_cache, v_cache, position_ids, attention_mask)


class TTSTextBlockWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, lm_hidden, k_cache, v_cache, position_ids, attention_mask):
        tts_lm = self.model.model.tts_language_model
        type_ids = torch.ones(
            (1, lm_hidden.shape[1]), dtype=torch.long, device=lm_hidden.device)
        x = lm_hidden + self.model.model.tts_input_types(type_ids)
        return decode_layers(
            tts_lm, x, k_cache, v_cache, position_ids, attention_mask)


def make_mask(past: int, query: int) -> torch.Tensor:
    mask = torch.full((1, 1, query, past + query), -10_000.0)
    for row in range(query):
        mask[:, :, row, : past + row + 1] = 0
    return mask


def quantize_and_save(ct, model, destination: Path) -> None:
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    compressed = linear_quantize_weights(
        model,
        config=OptimizationConfig(
            global_config=OpLinearQuantizerConfig(dtype="int8")),
    )
    compressed.save(str(destination))


def max_diff(torch_value: torch.Tensor, coreml_value) -> float:
    return float(np.max(np.abs(
        torch_value.detach().float().cpu().numpy() - coreml_value)))


def convert_one(ct, wrapper, lm, name: str, include_ids: bool) -> None:
    query = 5
    past = 16
    hidden = lm.config.hidden_size
    query_heads = lm.config.num_attention_heads
    kv_heads = lm.config.num_key_value_heads or query_heads
    head_dim = hidden // query_heads
    layers = len(lm.layers)
    max_t = min(lm.config.max_position_embeddings or 8192, 8192)
    ex_k = torch.randn(layers, 1, kv_heads, past, head_dim) * 0.05
    ex_v = torch.randn(layers, 1, kv_heads, past, head_dim) * 0.05
    ex_pos = torch.arange(past, past + query, dtype=torch.int32).reshape(1, query)
    ex_mask = make_mask(past, query)
    if include_ids:
        ex_primary = torch.randint(
            0, lm.config.vocab_size, (1, query), dtype=torch.int32)
        inputs = [
            ct.TensorType(name="input_ids", shape=(1, query), dtype=np.int32),
        ]
        primary_name = "input_ids"
    else:
        ex_primary = torch.randn(1, query, hidden)
        inputs = [ct.TensorType(name="lm_hidden", shape=(1, query, hidden))]
        primary_name = "lm_hidden"

    wrapper = wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (ex_primary, ex_k, ex_v, ex_pos, ex_mask), strict=False)
        py_outputs = wrapper(ex_primary, ex_k, ex_v, ex_pos, ex_mask)

    cache_range = ct.RangeDim(1, max_t, default=past)
    mask_range = ct.RangeDim(query + 1, max_t + query, default=past + query)
    inputs.extend([
        ct.TensorType(name="k_cache", shape=(layers, 1, kv_heads, cache_range, head_dim)),
        ct.TensorType(name="v_cache", shape=(layers, 1, kv_heads, cache_range, head_dim)),
        ct.TensorType(name="position_ids", shape=(1, query), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 1, query, mask_range)),
    ])
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="k_cache_out"),
            ct.TensorType(name="v_cache_out"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    values = {
        primary_name: ex_primary.numpy(),
        "k_cache": ex_k.numpy(),
        "v_cache": ex_v.numpy(),
        "position_ids": ex_pos.numpy(),
        "attention_mask": ex_mask.numpy(),
    }
    outputs = mlmodel.predict(values)
    for label, py_value, key in zip(
        ("hidden", "k", "v"),
        py_outputs,
        ("hidden_states", "k_cache_out", "v_cache_out"),
    ):
        print(f"[parity] {name} {label}: max|delta|={max_diff(py_value, outputs[key]):.3e}")

    destination = OUT_DIR / f"{name}.mlpackage"
    quantize_and_save(ct, mlmodel, destination)
    print(f"Saved {destination}")


def main() -> int:
    import coremltools as ct

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=("all", "text", "tts"), default="all")
    args = parser.parse_args()
    manifest = load_manifest()
    model = load_vibevoice()
    if args.models in ("all", "text"):
        text_lm = resolve(model, manifest["paths"]["text_lm"])
        convert_one(
            ct,
            TextBlockWrapper(text_lm),
            text_lm,
            "text_lm_decode_block5",
            include_ids=True,
        )
    if args.models in ("all", "tts"):
        tts_lm = model.model.tts_language_model
        convert_one(
            ct,
            TTSTextBlockWrapper(model),
            tts_lm,
            "tts_lm_decode_block5",
            include_ids=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
