"""
Phase E re-trace — convert the TTS LM correctly.

The earlier `02_convert_llm.py --tts` traced the TTS LM as a plain Qwen2 LM,
which is wrong: the real call path is `forward_tts_lm`, which

  1. Embeds `input_ids` to get `inputs_embeds`.
  2. Splices `lm_last_hidden_state` into the TAIL positions of `inputs_embeds`
     (overwrites the last K rows).
  3. Adds a TYPE embedding via `self.model.tts_input_types(tts_text_masks.long())`
     (1=text position, 0=speech position).
  4. Feeds the resulting `inputs_embeds` through `tts_language_model`.

So the CoreML graph must accept (input_ids, lm_last_hidden_state, tts_text_masks,
…kv…, position_ids) and do the tail-splice + type-add inside the traced module.

Emits:
  out/tts_lm_prefill.mlpackage   (overwrites the wrong one from 02_convert_llm.py)
  out/tts_lm_decode.mlpackage    (overwrites the wrong one)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, assert_parity, load_manifest, load_vibevoice  # noqa: E402


class TTSPrefillWrapper(torch.nn.Module):
    """Full-sequence forward for TTS LM prefill.

    Inputs:
      input_ids           (1, S) int32
      lm_hidden_padded    (1, S, H) float32  — lm_last_hidden_state zero-padded
                                               on the head so it's S-long.
      tail_mask           (1, S, 1) float32  — 1.0 at tail positions where the
                                               LM hidden should override the
                                               token embedding, 0.0 elsewhere.
      tts_text_masks      (1, S) int32       — 1=text, 0=speech (→ type embed).
    Outputs:
      hidden_states (1, S, H), k_cache (L,1,KVH,S,D), v_cache (L,1,KVH,S,D)

    The dynamic-slice tail splice caused an ios18.slice_by_index zero-shape
    bug at predict time (same failure mode as 02_convert_llm.py decode). The
    mask form is all static ops.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, input_ids, lm_hidden_padded, tail_mask, tts_text_masks):
        embed = self.m.model.tts_language_model.embed_tokens
        tok_embeds = embed(input_ids)
        inputs_embeds = tok_embeds * (1.0 - tail_mask) + lm_hidden_padded * tail_mask
        inputs_embeds = inputs_embeds + self.m.model.tts_input_types(tts_text_masks.long())

        out = self.m.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        kv = out.past_key_values
        if hasattr(kv, "to_legacy_cache"):
            kv = kv.to_legacy_cache()
        ks, vs = [], []
        for layer in kv:
            if layer is None or layer[0] is None:
                continue
            ks.append(layer[0])
            vs.append(layer[1])
        return hidden, torch.stack(ks, dim=0), torch.stack(vs, dim=0)


class TTSDecodeWrapper(torch.nn.Module):
    """Single-token TTS LM step with manual KV concat (mirrors DecodeWrapper
    from 02_convert_llm.py) and the tail-splice + type-add on the 1-length
    `inputs_embeds`.

    At decode time S=K=1, so the tail splice collapses to "replace the single
    embed slot with lm_last_hidden_state".
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(
        self,
        input_ids,            # (1, 1) int32
        lm_hidden_padded,     # (1, 1, H) float32 — already the full slot at decode
        tail_mask,            # (1, 1, 1) float32 — 1.0 to override token embed, 0.0 to keep it
        tts_text_masks,       # (1, 1) int32
        k_cache,              # (L, 1, KVH, T, D)
        v_cache,              # (L, 1, KVH, T, D)
        position_ids,         # (1, 1) int32
    ):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        tts_lm = self.m.model.tts_language_model
        tok_embeds = tts_lm.embed_tokens(input_ids)
        inputs_embeds = tok_embeds * (1.0 - tail_mask) + lm_hidden_padded * tail_mask
        inputs_embeds = inputs_embeds + self.m.model.tts_input_types(tts_text_masks.long())

        x = inputs_embeds
        cos, sin = tts_lm.rotary_emb(x, position_ids)

        new_ks, new_vs = [], []
        for i, layer in enumerate(tts_lm.layers):
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

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            k_full = torch.cat([k_cache[i], k], dim=2)
            v_full = torch.cat([v_cache[i], v], dim=2)
            new_ks.append(k_full)
            new_vs.append(v_full)

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

        x = tts_lm.norm(x)
        return x, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


def main() -> int:
    import coremltools as ct

    model = load_vibevoice()
    manifest = load_manifest()
    cfg = manifest["config"]  # text LM config; TTS LM has its own shape
    tts_lm = model.model.tts_language_model
    H = tts_lm.config.hidden_size
    Hq = tts_lm.config.num_attention_heads
    KVH = tts_lm.config.num_key_value_heads or Hq
    D = tts_lm.config.hidden_size // Hq
    L = len(tts_lm.layers)
    VOCAB = tts_lm.config.vocab_size
    MAX_T = min(tts_lm.config.max_position_embeddings or 8192, 8192)
    print(f"TTS LM: L={L} H={H} Hq={Hq} KVH={KVH} D={D} vocab={VOCAB} max_T={MAX_T}")

    # --- Prefill ------------------------------------------------------------
    pre = TTSPrefillWrapper(model).eval()
    S = 16
    ex_ids = torch.randint(0, VOCAB, (1, S), dtype=torch.int32)
    ex_lmp = torch.randn(1, S, H)
    ex_tail = torch.ones(1, S, 1)
    ex_tm = torch.ones(1, S, dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(pre, (ex_ids, ex_lmp, ex_tail, ex_tm), strict=False)

    ml_pre = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, MAX_T)), dtype=np.int32),
            ct.TensorType(name="lm_hidden_padded",
                          shape=(1, ct.RangeDim(1, MAX_T), H)),
            ct.TensorType(name="tail_mask",
                          shape=(1, ct.RangeDim(1, MAX_T), 1)),
            ct.TensorType(name="tts_text_masks",
                          shape=(1, ct.RangeDim(1, MAX_T)), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="k_cache"),
            ct.TensorType(name="v_cache"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_pre.save(str(OUT_DIR / "tts_lm_prefill.mlpackage"))

    with torch.no_grad():
        py_h, _, _ = pre(ex_ids, ex_lmp, ex_tail, ex_tm)
    ml_out = ml_pre.predict({
        "input_ids": ex_ids.numpy(),
        "lm_hidden_padded": ex_lmp.numpy(),
        "tail_mask": ex_tail.numpy(),
        "tts_text_masks": ex_tm.numpy(),
    })
    assert_parity(py_h, ml_out["hidden_states"], tol=5e-3, label="tts_lm_prefill hidden")

    # --- Decode -------------------------------------------------------------
    dec = TTSDecodeWrapper(model).eval()
    ex_id = torch.randint(0, VOCAB, (1, 1), dtype=torch.int32)
    ex_lmp1 = torch.randn(1, 1, H)
    ex_tail1 = torch.ones(1, 1, 1)
    ex_tm1 = torch.ones(1, 1, dtype=torch.int32)
    ex_k = torch.zeros(L, 1, KVH, 16, D)
    ex_v = torch.zeros(L, 1, KVH, 16, D)
    ex_pos = torch.tensor([[16]], dtype=torch.int32)
    with torch.no_grad():
        traced_d = torch.jit.trace(
            dec, (ex_id, ex_lmp1, ex_tail1, ex_tm1, ex_k, ex_v, ex_pos), strict=False
        )
    ml_dec = ct.convert(
        traced_d,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="lm_hidden_padded", shape=(1, 1, H)),
            ct.TensorType(name="tail_mask", shape=(1, 1, 1)),
            ct.TensorType(name="tts_text_masks", shape=(1, 1), dtype=np.int32),
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
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_dec.save(str(OUT_DIR / "tts_lm_decode.mlpackage"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
