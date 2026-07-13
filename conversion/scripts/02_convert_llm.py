"""
Phase B — Convert the Qwen2.5-0.5B text LM (and optionally the TTS LM) to
CoreML as two artifacts each: prefill + decode, with a stateful KV cache.

Emits: out/text_lm_prefill.mlpackage, out/text_lm_decode.mlpackage
       (and tts_lm_* if --tts is passed)

This script is the hardest of the four. Expect to iterate. Known pitfalls
(see docs/conversion-notes.md):
  - torch.arange always returns int32 under coremltools; add explicit casts.
  - RoPE mixes int and fp — rewrite to keep dtypes consistent.
  - Stateful KV needs ct.StateType + ct.target.iOS18.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, assert_parity, coreml_compute_precision, load_manifest, load_vibevoice, parity_tolerance, resolve  # noqa: E402


class PrefillWrapper(torch.nn.Module):
    """Full-sequence forward: input_ids → hidden_states + initial K/V per layer.

    Returning raw K/V tensors (not a HF Cache object) because coremltools
    traces concrete tensors, not opaque Python objects.
    """

    def __init__(self, lm: torch.nn.Module):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.lm(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        kv = out.past_key_values
        ks, vs = [], []
        # HF 4.57 returns either a DynamicCache or a legacy tuple-of-tuples.
        if hasattr(kv, "to_legacy_cache"):
            kv = kv.to_legacy_cache()
        for layer in kv:
            if layer is None or layer[0] is None:
                continue
            ks.append(layer[0])
            vs.append(layer[1])
        return (hidden, torch.stack(ks, dim=0), torch.stack(vs, dim=0))


class DecodeWrapper(torch.nn.Module):
    """Single-token step with a pre-populated KV cache, cache concat done
    manually so HF's DynamicCache internals don't trip the tracer.

    We call the LM layer-by-layer, mirroring Qwen2Model.forward but skipping
    the HF Cache object entirely.
    """

    def __init__(self, lm: torch.nn.Module):
        super().__init__()
        self.lm = lm

    def forward(
        self,
        input_ids: torch.Tensor,    # (1, 1) int32
        k_cache: torch.Tensor,      # (L, 1, KVH, T, D)
        v_cache: torch.Tensor,      # (L, 1, KVH, T, D)
        position_ids: torch.Tensor, # (1, 1) int32
    ):
        x = self.lm.embed_tokens(input_ids)
        # Qwen2 RoPE lives on self.lm.rotary_emb (HF 4.57). Compute cos/sin once.
        cos, sin = self.lm.rotary_emb(x, position_ids)

        new_ks, new_vs = [], []
        for i, layer in enumerate(self.lm.layers):
            residual = x
            h = layer.input_layernorm(x)

            attn = layer.self_attn
            q = attn.q_proj(h)
            k = attn.k_proj(h)
            v = attn.v_proj(h)

            B, T, _ = q.shape
            Hq = attn.config.num_attention_heads
            Hkv = attn.config.num_key_value_heads
            D = attn.head_dim
            q = q.view(B, T, Hq, D).transpose(1, 2)
            k = k.view(B, T, Hkv, D).transpose(1, 2)
            v = v.view(B, T, Hkv, D).transpose(1, 2)

            # Apply RoPE
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Append to cache
            k_full = torch.cat([k_cache[i], k], dim=2)
            v_full = torch.cat([v_cache[i], v], dim=2)
            new_ks.append(k_full)
            new_vs.append(v_full)

            # GQA expand
            rep = Hq // Hkv
            k_exp = k_full.repeat_interleave(rep, dim=1)
            v_exp = v_full.repeat_interleave(rep, dim=1)

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, T, Hq * D)
            attn_out = attn.o_proj(attn_out)

            x = residual + attn_out
            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)

        x = self.lm.norm(x)
        return x, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


def convert(lm_path: str, out_prefix: str, manifest: dict) -> None:
    import coremltools as ct

    print(f"\n=== Converting {lm_path} → {out_prefix}_{{prefill,decode}}.mlpackage ===")
    model = load_vibevoice()
    lm = resolve(model, lm_path)

    cfg = manifest["config"]
    H = cfg["num_attention_heads"]
    KVH = cfg["num_key_value_heads"] or H
    D = cfg["head_dim"]
    # config.num_hidden_layers is aspirational (24 for text LM). The realtime
    # checkpoint instantiates only len(lm.layers) layers (4 text / 20 tts).
    L = len(lm.layers)
    MAX_T = min(cfg["max_position"] or 8192, 8192)
    print(f"  actual layers: {L} (config says {cfg['num_hidden_layers']})")

    # --- Prefill -------------------------------------------------------------
    prefill = PrefillWrapper(lm).eval()
    example_ids = torch.randint(0, cfg["vocab_size"], (1, 16), dtype=torch.int32)
    with torch.no_grad():
        traced_prefill = torch.jit.trace(prefill, example_ids, strict=False)

    ml_prefill = ct.convert(
        traced_prefill,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape((1, ct.RangeDim(1, MAX_T))),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="k_cache"),
            ct.TensorType(name="v_cache"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=coreml_compute_precision(ct),
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_prefill.save(str(OUT_DIR / f"{out_prefix}_prefill.mlpackage"))

    # Prefill parity gate — don't let later decode bugs mask this.
    with torch.no_grad():
        py_hidden, _, _ = prefill(example_ids)
    ml_out = ml_prefill.predict({"input_ids": example_ids.numpy()})
    # 2e-3 tol: 195M-param LM traced via SDPA shows ~1e-3 drift on last layer.
    assert_parity(
        py_hidden,
        ml_out["hidden_states"],
        tol=parity_tolerance(float32=2e-3, float16=2e-2),
        label=f"{out_prefix} prefill hidden",
    )

    # --- Decode --------------------------------------------------------------
    decode = DecodeWrapper(lm).eval()
    example_k = torch.zeros(L, 1, KVH, 16, D)
    example_v = torch.zeros(L, 1, KVH, 16, D)
    example_pos = torch.tensor([[16]], dtype=torch.int32)
    example_id = torch.randint(0, cfg["vocab_size"], (1, 1), dtype=torch.int32)
    with torch.no_grad():
        traced_decode = torch.jit.trace(
            decode, (example_id, example_k, example_v, example_pos), strict=False
        )

    ml_decode = ct.convert(
        traced_decode,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="k_cache", shape=(L, 1, KVH, ct.RangeDim(1, MAX_T), D)),
            ct.TensorType(name="v_cache", shape=(L, 1, KVH, ct.RangeDim(1, MAX_T), D)),
            ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="k_cache_out"),
            ct.TensorType(name="v_cache_out"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=coreml_compute_precision(ct),
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_decode.save(str(OUT_DIR / f"{out_prefix}_decode.mlpackage"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true", help="Also convert the TTS LM.")
    args = ap.parse_args()

    manifest = load_manifest()
    convert(manifest["paths"]["text_lm"], "text_lm", manifest)
    if args.tts and manifest["paths"].get("tts_lm"):
        convert(manifest["paths"]["tts_lm"], "tts_lm", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
