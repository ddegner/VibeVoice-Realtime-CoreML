"""
Generate a PyTorch reference trace for a *single* (text, voice, seed) triple
and dump every intermediate Swift needs to match to localize a divergence.

Motivation:
  - Audio A/B tests (current state of the union) show some seeds stutter in
    Swift but never in PyTorch. Likely culprits are (a) my DPM-Solver++
    scheduler, (b) my TTS-LM text-window feeding, (c) my acoustic-connector /
    feedback path. This script gives us a *ground truth* tape so a matching
    Swift diagnostic can `assert |Δ| < tol` at every stage and tell us which
    is wrong.

Outputs under `out/ref/<tag>/`:
  meta.json                 { text, voice, seed, steps, guidance,
                              tts_lm_pos_start, speech_frames_dumped }
  text_tokens.npy           int32  (T_text,)                      tokenized text
  noise_frame{f}.npy        float32(64,)                           initial noise seeded from seed+frame
  vcond_frame{f}_step{s}.npy   float32(64,)                        CFG pos branch
  vuncond_frame{f}_step{s}.npy float32(64,)                        CFG neg branch
  v_frame{f}_step{s}.npy    float32(64,)                           CFG-blended
  x0_frame{f}_step{s}.npy   float32(64,)                           predicted x0 (v→x0)
  sample_frame{f}_step{s}.npy float32(64,)                         x_t after step
  latent_frame{f}.npy       float32(64,)                           final clean latent
  posLast_frame{f}.npy      float32(896,)                          TTS LM cond fed to diffusion
  negLast_frame{f}.npy      float32(896,)                          TTS LM uncond fed to diffusion
  audio.wav                 24 kHz float32 PCM (whole utterance)
  tts_lm_first_window.npy   float32(5, 896)                        posLast after text window

Usage:
  python scripts/10_pytorch_reference_dump.py \
      --text "The quick brown fox jumps over the lazy dog." \
      --voice en-Frank_man --seed 42 --steps 5 --guidance 1.5 \
      --frames 3 --tag fox_s42_st5

Parity expectations in a future Swift diagnostic:
  - text_tokens: must match Swift's tokenizer exactly
  - posLast_frame0, negLast_frame0: match after first text window
  - noise_frame0: matches iff Swift uses the same PRNG (SplitMix64 vs torch)
      -> if they differ, use this dump directly as input to Swift
  - vcond / vuncond: must match if noise + posLast/negLast match
  - x0 / sample: must match the DPM-Solver math
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="en-Frank_man")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=3,
                    help="How many speech frames to capture full traces for.")
    ap.add_argument("--tag", required=True, help="Subdir under out/ref/")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = OUT_DIR / "ref" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing reference dump to {out_dir}")

    # Lazy imports of upstream pieces
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference as VVSInfer,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    # ---- Load ------------------------------------------------------------
    device = "cpu"  # reference stays on CPU for determinism
    dtype = torch.float32
    model = VVSInfer.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    processor = VibeVoiceStreamingProcessor.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
    model.set_ddpm_inference_steps(args.steps)

    # ---- Load voice prompt ----------------------------------------------
    voices_dir = Path(__file__).resolve().parents[1] \
        / "third_party" / "VibeVoice" / "demo" / "voices" / "streaming_model"
    voice_path = voices_dir / f"{args.voice}.pt"
    all_prefilled_outputs = torch.load(str(voice_path), map_location=device, weights_only=False)

    # ---- Build generate() inputs via the same processor the demo uses ----
    # Upstream's streaming processor builds:
    #   input_ids / attention_mask         — pseudo pad, length = lm prefill cache len
    #   tts_lm_input_ids / tts_lm_attention_mask — pseudo pad, length = tts_lm prefill cache len
    #   tts_text_ids                       — the actual text tokens to stream
    # The trailing "\n" is significant — matches the demo behavior.
    inputs = processor.process_input_with_cached_prompt(
        text=args.text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    text_tokens = processor.tokenizer.encode(args.text.strip() + "\n", add_special_tokens=False)
    np.save(out_dir / "text_tokens.npy", np.asarray(text_tokens, dtype=np.int32))
    print(f"Tokenized to {len(text_tokens)} ids: {text_tokens[:20]}...")

    # ---- Instrument the diffusion sampler -------------------------------
    # We wrap `sample_speech_tokens` to (a) seed noise deterministically per
    # frame and (b) dump every intermediate. Upstream's implementation is in
    # the file itself — we replicate its algorithm here to stay in control.
    frame_counter = {"i": 0}
    dumps_budget = args.frames

    # Capture posLast / negLast by hooking forward_tts_lm's return.
    posLast_history: list[np.ndarray] = []
    negLast_history: list[np.ndarray] = []

    # Capture the first text-window's TTS-LM hidden output
    first_tts_window_hidden: dict = {"value": None}

    orig_forward_tts_lm = model.forward_tts_lm
    tts_call_counter = {"i": 0}
    # Speech-window TTS calls are at lines 822 (pos) and 841 (neg). Text-window
    # call is at 764 (pos only). They're interleaved, so we track by call order:
    #   call 0: text-window pos
    #   call 1..6: speech-window pos (one per frame in the window)
    #   call 2..6 neg follows each pos in the same speech step
    # We'll just capture the first text-window call specifically and then
    # bookkeep posLast / negLast on every speech-step call.

    def _to_np(t):
        """Convert a tensor to fp32 numpy — bfloat16 isn't numpy-native."""
        return t.detach().float().cpu().numpy()

    def traced_forward_tts_lm(*a, **kw):
        out = orig_forward_tts_lm(*a, **kw)
        i = tts_call_counter["i"]
        tts_call_counter["i"] += 1
        if i == 0:
            # First text window — dump the full 5-token hidden state
            first_tts_window_hidden["value"] = _to_np(out.last_hidden_state)
        return out

    model.forward_tts_lm = traced_forward_tts_lm

    # For posLast/negLast and speech-window diffusion intermediates we
    # replace sample_speech_tokens wholesale (reimplemented w/ dumps).
    noise_scheduler = model.noise_scheduler
    ddpm_inference_steps = args.steps

    def sample_speech_tokens_instrumented(condition, neg_condition, cfg_scale=3.0):
        """Byte-for-byte port of upstream's sample_speech_tokens() with per-step dumps."""
        frame_idx = frame_counter["i"]
        frame_counter["i"] += 1
        should_dump = frame_idx < dumps_budget

        # Dump the TTS-LM conditioning
        cond_np = _to_np(condition)[0]
        neg_np = _to_np(neg_condition)[0]
        posLast_history.append(cond_np)
        negLast_history.append(neg_np)
        if should_dump:
            np.save(out_dir / f"posLast_frame{frame_idx}.npy", cond_np)
            np.save(out_dir / f"negLast_frame{frame_idx}.npy", neg_np)

        # --- exact mirror of upstream's sample_speech_tokens ---
        noise_scheduler.set_timesteps(ddpm_inference_steps)
        combined_cond = torch.cat([condition, neg_condition], dim=0).to(
            model.model.prediction_head.device
        )

        # Seeded initial noise (seed = args.seed * 10_000 + frame_idx)
        gen = torch.Generator(device=condition.device)
        gen.manual_seed(args.seed * 10_000 + frame_idx)
        speech = torch.randn(
            combined_cond.shape[0], model.config.acoustic_vae_dim,
            generator=gen, device=condition.device, dtype=combined_cond.dtype,
        )
        if should_dump:
            np.save(out_dir / f"noise_frame{frame_idx}.npy", _to_np(speech[0]))

        for step_idx, t in enumerate(noise_scheduler.timesteps):
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = model.model.prediction_head(
                combined,
                t.repeat(combined.shape[0]).to(combined),
                condition=combined_cond,
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps_cat = torch.cat([half_eps, half_eps], dim=0)

            # Compute x0 explicitly (v-prediction: x0 = alpha*sample - sigma*v).
            # Derive scalar alpha/sigma from the scheduler's alphas_cumprod @ t —
            # avoids whatever shape drama is going on in scheduler.sigmas.
            alpha_bar = float(noise_scheduler.alphas_cumprod[t.item()].item())
            alpha_t = math.sqrt(alpha_bar)
            sigma_t = math.sqrt(max(0.0, 1.0 - alpha_bar))
            x0 = alpha_t * half - sigma_t * half_eps

            new_speech = noise_scheduler.step(eps_cat, t, speech).prev_sample

            if should_dump:
                np.save(out_dir / f"vcond_frame{frame_idx}_step{step_idx}.npy", _to_np(cond_eps[0]))
                np.save(out_dir / f"vuncond_frame{frame_idx}_step{step_idx}.npy", _to_np(uncond_eps[0]))
                np.save(out_dir / f"v_frame{frame_idx}_step{step_idx}.npy", _to_np(half_eps[0]))
                np.save(out_dir / f"x0_frame{frame_idx}_step{step_idx}.npy", _to_np(x0[0]))
                np.save(out_dir / f"sample_frame{frame_idx}_step{step_idx}.npy", _to_np(new_speech[0]))

            speech = new_speech

        final = speech[: len(speech) // 2]  # pos branch
        if should_dump:
            np.save(out_dir / f"latent_frame{frame_idx}.npy", _to_np(final[0]))
        return final

    model.sample_speech_tokens = sample_speech_tokens_instrumented

    # ---- Run generation --------------------------------------------------
    print(f"\nGenerating with text='{args.text}' voice={args.voice} "
          f"seed={args.seed} steps={args.steps} cfg={args.guidance}")
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=None,
            tokenizer=processor.tokenizer,
            all_prefilled_outputs=all_prefilled_outputs,
            cfg_scale=args.guidance,
            generation_config={"do_sample": False},
            return_speech=True,
            show_progress_bar=False,
            verbose=False,
        )

    audio_list = result.speech_outputs
    if audio_list and audio_list[0] is not None:
        audio = _to_np(audio_list[0].squeeze()).astype(np.float32)
        sf.write(str(out_dir / "audio.wav"), audio, 24_000)
        print(f"Audio: {len(audio)} samples ({len(audio)/24000:.2f}s) → audio.wav")

    # ---- Dump collected artifacts ---------------------------------------
    if first_tts_window_hidden["value"] is not None:
        h = first_tts_window_hidden["value"]
        np.save(out_dir / "tts_lm_first_window.npy", h[0])  # (T, 896)
        print(f"First text-window TTS-LM hidden: {h.shape}")

    meta = {
        "text": args.text,
        "voice": args.voice,
        "seed": args.seed,
        "steps": args.steps,
        "guidance": args.guidance,
        "n_tokens": len(text_tokens),
        "frames_dumped": min(args.frames, frame_counter["i"]),
        "total_frames": frame_counter["i"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Reference tape at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
