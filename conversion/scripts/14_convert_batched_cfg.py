#!/usr/bin/env python3
"""Export batch-2 CFG models for the speed/quality benchmark.

The TTS model is specialized for speech-frame decode: both CFG branches share
the same acoustic embedding, while their KV caches, positions, and attention
masks remain independent. The diffusion model similarly evaluates cond/uncond
in one batch. Outputs are weight-quantized to int8 to match the shipping model
directory; activations and compute remain float32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, load_manifest, load_vibevoice, resolve  # noqa: E402


class TTSSpeechCFGWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        lm_hidden: torch.Tensor,       # (1, 1, H), shared acoustic embedding
        k_cache: torch.Tensor,         # (L, 2, KVH, T, D), padded/masked
        v_cache: torch.Tensor,
        position_ids: torch.Tensor,    # (2, 1), logical branch positions
        attention_mask: torch.Tensor,  # (2, 1, 1, T+1), additive mask
    ):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        tts_lm = self.model.model.tts_language_model
        type_ids = torch.zeros((2, 1), dtype=torch.long, device=lm_hidden.device)
        x = torch.cat([lm_hidden, lm_hidden], dim=0)
        x = x + self.model.model.tts_input_types(type_ids)
        cos, sin = tts_lm.rotary_emb(x, position_ids)

        new_ks, new_vs = [], []
        for i, layer in enumerate(tts_lm.layers):
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

            k_full = torch.cat([k_cache[i], k], dim=2)
            v_full = torch.cat([v_cache[i], v], dim=2)
            new_ks.append(k_full)
            new_vs.append(v_full)

            repeats = query_heads // kv_heads
            k_expanded = k_full.repeat_interleave(repeats, dim=1)
            v_expanded = v_full.repeat_interleave(repeats, dim=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attention_mask,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).reshape(
                batch, query_len, query_heads * head_dim)
            x = residual + attn.o_proj(attn_out)
            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)

        x = tts_lm.norm(x)
        return x, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


class DiffusionCFGWrapper(torch.nn.Module):
    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head

    def forward(self, noisy_latent, timestep, conditioning, unconditioning):
        latent_batch = torch.cat([noisy_latent, noisy_latent], dim=0)
        timestep_batch = torch.cat([timestep, timestep], dim=0)
        conditioning_batch = torch.cat([conditioning, unconditioning], dim=0)
        prediction = self.head(latent_batch, timestep_batch, conditioning_batch)
        return prediction[0:1], prediction[1:2]


def quantize_and_save(ct, model, path: Path) -> None:
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(dtype="int8"))
    compressed = linear_quantize_weights(model, config=config)
    compressed.save(str(path))


def max_diff(torch_value: torch.Tensor, coreml_value) -> float:
    a = torch_value.detach().float().cpu().numpy()
    return float(np.max(np.abs(a - coreml_value)))


def convert_tts(ct, model) -> None:
    tts_lm = model.model.tts_language_model
    hidden = tts_lm.config.hidden_size
    query_heads = tts_lm.config.num_attention_heads
    kv_heads = tts_lm.config.num_key_value_heads or query_heads
    head_dim = hidden // query_heads
    layers = len(tts_lm.layers)
    max_t = min(tts_lm.config.max_position_embeddings or 8192, 8192)

    wrapper = TTSSpeechCFGWrapper(model).eval()
    cache_len = 16
    ex_hidden = torch.randn(1, 1, hidden)
    ex_k = torch.randn(layers, 2, kv_heads, cache_len, head_dim) * 0.05
    ex_v = torch.randn(layers, 2, kv_heads, cache_len, head_dim) * 0.05
    ex_pos = torch.tensor([[16], [3]], dtype=torch.int32)
    ex_mask = torch.zeros(2, 1, 1, cache_len + 1)
    ex_mask[1, :, :, :13] = -10_000

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (ex_hidden, ex_k, ex_v, ex_pos, ex_mask), strict=False)
        py_outputs = wrapper(ex_hidden, ex_k, ex_v, ex_pos, ex_mask)

    cache_range = ct.RangeDim(1, max_t, default=cache_len)
    mask_range = ct.RangeDim(2, max_t + 1, default=cache_len + 1)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="lm_hidden", shape=(1, 1, hidden)),
            ct.TensorType(name="k_cache", shape=(layers, 2, kv_heads, cache_range, head_dim)),
            ct.TensorType(name="v_cache", shape=(layers, 2, kv_heads, cache_range, head_dim)),
            ct.TensorType(name="position_ids", shape=(2, 1), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(2, 1, 1, mask_range)),
        ],
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
    ml_outputs = mlmodel.predict({
        "lm_hidden": ex_hidden.numpy(),
        "k_cache": ex_k.numpy(),
        "v_cache": ex_v.numpy(),
        "position_ids": ex_pos.numpy(),
        "attention_mask": ex_mask.numpy(),
    })
    for label, py_value, key in zip(
        ("hidden", "k", "v"),
        py_outputs,
        ("hidden_states", "k_cache_out", "v_cache_out"),
    ):
        print(f"[parity] tts cfg2 {label}: max|delta|={max_diff(py_value, ml_outputs[key]):.3e}")

    destination = OUT_DIR / "tts_lm_decode_cfg2.mlpackage"
    quantize_and_save(ct, mlmodel, destination)
    print(f"Saved {destination}")


def convert_diffusion(ct, model, manifest) -> None:
    head = resolve(model, manifest["paths"]["diffusion_head"])
    latent_dim = getattr(head.config, "latent_size", 64)
    hidden_dim = getattr(head.config, "hidden_size", manifest["config"]["hidden_size"])
    wrapper = DiffusionCFGWrapper(head).eval()
    ex_latent = torch.randn(1, 1, latent_dim)
    ex_timestep = torch.tensor([500.0], dtype=torch.float32)
    ex_cond = torch.randn(1, 1, hidden_dim)
    ex_uncond = torch.randn(1, 1, hidden_dim)
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (ex_latent, ex_timestep, ex_cond, ex_uncond), strict=False)
        py_cond, py_uncond = wrapper(ex_latent, ex_timestep, ex_cond, ex_uncond)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="noisy_latent", shape=(1, 1, latent_dim)),
            ct.TensorType(name="timestep", shape=(1,)),
            ct.TensorType(name="conditioning", shape=(1, 1, hidden_dim)),
            ct.TensorType(name="unconditioning", shape=(1, 1, hidden_dim)),
        ],
        outputs=[
            ct.TensorType(name="predicted_noise_cond"),
            ct.TensorType(name="predicted_noise_uncond"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    outputs = mlmodel.predict({
        "noisy_latent": ex_latent.numpy(),
        "timestep": ex_timestep.numpy(),
        "conditioning": ex_cond.numpy(),
        "unconditioning": ex_uncond.numpy(),
    })
    print(
        "[parity] diffusion cfg2 cond: "
        f"max|delta|={max_diff(py_cond, outputs['predicted_noise_cond']):.3e}")
    print(
        "[parity] diffusion cfg2 uncond: "
        f"max|delta|={max_diff(py_uncond, outputs['predicted_noise_uncond']):.3e}")

    destination = OUT_DIR / "diffusion_head_cfg2.mlpackage"
    quantize_and_save(ct, mlmodel, destination)
    print(f"Saved {destination}")


def main() -> int:
    import coremltools as ct

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        choices=("all", "tts", "diffusion"),
        default="all",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    model = load_vibevoice()
    if args.models in ("all", "tts"):
        convert_tts(ct, model)
    if args.models in ("all", "diffusion"):
        convert_diffusion(ct, model, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
