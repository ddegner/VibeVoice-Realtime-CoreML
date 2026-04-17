"""
Run the *CoreML* pipeline (the .mlpackages we'd ship to Swift) end-to-end in
Python, using the same scheduler math our Swift `DiffusionRunner` uses. Dumps
matching intermediates so `12_compare_dumps.py` can localize divergence
between:

  A) PyTorch reference       (10_pytorch_reference_dump.py)
  B) CoreML-via-Python       (this script)        <- isolates conversion
  C) Swift CLI                                    <- isolates orchestration

If A vs B differ:  the .mlpackage conversion is wrong.
If B vs C differ:  Swift orchestration / scheduler / RNG is wrong.
If A ≈ B ≈ C:      either we're hearing minor stochastic differences, or the
                   PyTorch reference itself is not the ground truth (e.g.
                   running upstream with a different random draw).

This script intentionally re-implements the scheduler step the *same way*
DiffusionRunner.swift does, so we can verify that implementation on its own
by comparing to HF's scheduler.step() output (run in 10_pytorch_...).

Outputs under out/ref/<tag>/coreml/ — same filenames as the PyTorch dump, so
the comparison tool can zip them pairwise.

Usage:
  python scripts/11_coreml_trace_dump.py \
      --text "The quick brown fox jumps over the lazy dog." \
      --voice en-Frank_man --seed 42 --steps 5 --guidance 1.5 \
      --frames 3 --tag fox_s42_st5 [--use-ref-noise]

  --use-ref-noise: load noise_frame{f}.npy from the PyTorch dump dir so initial
                   noise matches exactly. Required for per-step value parity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402

TEXT_WINDOW = 5
SPEECH_WINDOW = 6
VAE_T = 16
VAE_SAMPLES_PER_FRAME = 3200


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="en-Frank_man")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--use-ref-noise", action="store_true")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--eos-threshold", type=float, default=0.5)
    return ap.parse_args()


# ---- Match DiffusionRunner.swift's schedule construction ----------------
def build_schedule(steps: int):
    """Mirror DiffusionRunner.swift after the alpha_bar off-by-one fix."""
    N = 1000
    s = 0.008

    def alpha_bar_fn(u):
        return math.cos((u + s) / (1.0 + s) * math.pi / 2) ** 2

    betas = []
    for t in range(N):
        t1 = t / N
        t2 = (t + 1) / N
        betas.append(min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
    alphas_cumprod = np.cumprod(1.0 - np.array(betas))

    # Timesteps: np.linspace(0, 999, steps+1).round()[::-1][:-1]
    timesteps = np.linspace(0, N - 1, steps + 1).round().astype(int)[::-1][:-1]

    entries = []
    for t in timesteps:
        a_bar = float(alphas_cumprod[t])
        alpha_t = math.sqrt(a_bar)
        sigma_t = math.sqrt(1.0 - a_bar)
        lambda_t = math.log(alpha_t / sigma_t)
        entries.append({"t": int(t), "alpha": alpha_t, "sigma": sigma_t, "lambda": lambda_t})
    return entries


def dpm_first_order(sample, x0, sigma_s, alpha_t, sigma_t, h):
    if sigma_t < 1e-12:
        return float(alpha_t) * x0
    ratio = sigma_t / sigma_s
    coeff = alpha_t * (1.0 - math.exp(-h))
    return ratio * sample + coeff * x0


def dpm_second_order(sample, x0, prev_x0, sigma_s, alpha_t, sigma_t, h, r0):
    if sigma_t < 1e-12:
        return float(alpha_t) * x0
    ratio = sigma_t / sigma_s
    expmh_m1 = math.exp(-h) - 1.0
    aCoeff = alpha_t * expmh_m1
    invR0 = 1.0 / r0
    D1 = invR0 * (x0 - prev_x0)
    return ratio * sample - aCoeff * x0 - 0.5 * aCoeff * D1


def v_to_x0(sample, v, alpha_t, sigma_t):
    return alpha_t * sample - sigma_t * v


# ---- CoreML model wrappers ----------------------------------------------
def load_coreml_models(models_dir: Path):
    import coremltools as ct

    def _load(name):
        pkg = models_dir / f"{name}.mlpackage"
        if not pkg.exists():
            raise FileNotFoundError(pkg)
        return ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    return {
        "text_decode":      _load("text_lm_decode"),
        "tts_decode":       _load("tts_lm_decode"),
        "diffusion":        _load("diffusion_head"),
        "vae":              _load("vae_decoder"),
        "acoustic_connector": _load("acoustic_connector"),
        "tts_eos":          _load("tts_eos_classifier"),
    }


def load_voice(voice_dir: Path):
    meta = json.loads((voice_dir / "metadata.json").read_text())
    def _load(name):
        return np.load(voice_dir / f"{name}.npy")
    return {
        "lm":         {"hidden": _load("lm_hidden"),     "k": _load("lm_k_cache"),     "v": _load("lm_v_cache"),     "len": meta["lm_len"]},
        "tts_lm":     {"hidden": _load("tts_lm_hidden"), "k": _load("tts_lm_k_cache"), "v": _load("tts_lm_v_cache"), "len": meta["tts_lm_len"]},
        "neg_tts_lm": {"hidden": _load("neg_tts_lm_hidden"), "k": _load("neg_tts_lm_k_cache"), "v": _load("neg_tts_lm_v_cache"), "len": meta["neg_tts_lm_len"]},
    }


def text_decode_step(model, tok, k_cache, v_cache, pos_id):
    out = model.predict({
        "input_ids": np.array([[tok]], dtype=np.int32),
        "k_cache": k_cache.astype(np.float32),
        "v_cache": v_cache.astype(np.float32),
        "position_ids": np.array([[pos_id]], dtype=np.int32),
    })
    return out["hidden_states"], out["k_cache_out"], out["v_cache_out"]


def tts_decode_step(model, tok, lm_hidden, tail_mask, text_mask, k_cache, v_cache, pos_id):
    out = model.predict({
        "input_ids": np.array([[tok]], dtype=np.int32),
        "lm_hidden_padded": lm_hidden.astype(np.float32),
        "tail_mask": np.array([[[tail_mask]]], dtype=np.float32),
        "tts_text_masks": np.array([[text_mask]], dtype=np.int32),
        "k_cache": k_cache.astype(np.float32),
        "v_cache": v_cache.astype(np.float32),
        "position_ids": np.array([[pos_id]], dtype=np.int32),
    })
    return out["hidden_states"], out["k_cache_out"], out["v_cache_out"]


def diffusion_predict(model, latent, timestep, cond):
    """latent: (1,1,64), timestep: (1,), cond: (1,1,896) → predicted_noise (1,1,64)"""
    out = model.predict({
        "noisy_latent": latent.astype(np.float32),
        "timestep": np.array([timestep], dtype=np.float32),
        "conditioning": cond.astype(np.float32),
    })
    return out["predicted_noise"]


def denoise_frame(diff_model, cond896, uncond896, schedule, steps, guidance, init_noise,
                  frame_idx, dump_dir, should_dump, rng):
    """Mirror DiffusionRunner.swift's denoiseFrame() exactly. Dumps per-step intermediates."""
    sample = init_noise.copy()  # shape (64,)
    cond3 = cond896.reshape(1, 1, 896).astype(np.float32)
    uncond3 = uncond896.reshape(1, 1, 896).astype(np.float32)

    prev_x0 = None
    order_used = 0

    if should_dump:
        np.save(dump_dir / f"noise_frame{frame_idx}.npy", sample.astype(np.float32))

    for i in range(steps):
        entry = schedule[i]
        latent = sample.reshape(1, 1, 64)
        v_cond = diffusion_predict(diff_model, latent, entry["t"], cond3).flatten()
        v_uncond = diffusion_predict(diff_model, latent, entry["t"], uncond3).flatten()
        v = v_uncond + guidance * (v_cond - v_uncond)
        x0 = v_to_x0(sample, v, entry["alpha"], entry["sigma"])

        if i + 1 < steps:
            target = schedule[i + 1]
        else:
            target = {"alpha": 1.0, "sigma": 0.0, "lambda": float("inf")}

        this_order = 1 if (i == 0 or i == steps - 1) else min(2, order_used + 1)
        h = target["lambda"] - entry["lambda"]

        if this_order == 1:
            new_sample = dpm_first_order(sample, x0, entry["sigma"],
                                         target["alpha"], target["sigma"], h)
        else:
            r0 = (entry["lambda"] - schedule[i - 1]["lambda"]) / h
            new_sample = dpm_second_order(sample, x0, prev_x0, entry["sigma"],
                                          target["alpha"], target["sigma"], h, r0)

        if should_dump:
            np.save(dump_dir / f"vcond_frame{frame_idx}_step{i}.npy", v_cond.astype(np.float32))
            np.save(dump_dir / f"vuncond_frame{frame_idx}_step{i}.npy", v_uncond.astype(np.float32))
            np.save(dump_dir / f"v_frame{frame_idx}_step{i}.npy", v.astype(np.float32))
            np.save(dump_dir / f"x0_frame{frame_idx}_step{i}.npy", x0.astype(np.float32))
            np.save(dump_dir / f"sample_frame{frame_idx}_step{i}.npy",
                    np.asarray(new_sample, dtype=np.float32))

        prev_x0 = x0
        order_used = this_order
        sample = new_sample

    if should_dump:
        np.save(dump_dir / f"latent_frame{frame_idx}.npy", sample.astype(np.float32))
    return sample


class SplitMix64:
    """Mirror DiffusionRunner.swift SplitMix64 for bit-identical noise to Swift."""
    GOLDEN = 0x9E3779B97F4A7C15
    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next(self):
        self.state = (self.state + self.GOLDEN) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (z ^ (z >> 31)) & self.MASK

    def next_unit(self):
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def randn(self, n: int):
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            u1 = max(np.finfo(np.float32).tiny, self.next_unit())
            u2 = self.next_unit()
            out[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return out


def main() -> int:
    args = parse_args()
    ref_dir = OUT_DIR / "ref" / args.tag
    dump_dir = ref_dir / "coreml"
    dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing CoreML trace dump to {dump_dir}")

    models = load_coreml_models(OUT_DIR)
    speech_const = json.loads((OUT_DIR / "speech_constants.json").read_text())
    voice = load_voice(OUT_DIR / "voices" / args.voice)

    # Tokenize. Match VibeVoiceStreamingProcessor: strip + append "\n" before
    # encoding. Swift currently skips the trailing "\n" — if that ever matters
    # the text_tokens comparison in script 12 will surface it.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(OUT_DIR / "tokenizer"))
    tokens = tok.encode(args.text.strip() + "\n", add_special_tokens=False)
    np.save(dump_dir / "text_tokens.npy", np.asarray(tokens, dtype=np.int32))
    print(f"Tokens ({len(tokens)}): {tokens[:20]}...")

    schedule = build_schedule(args.steps)
    print(f"Schedule (first t={schedule[0]['t']}, alpha={schedule[0]['alpha']:.6f}, "
          f"sigma={schedule[0]['sigma']:.6f}, lambda={schedule[0]['lambda']:+.4f})")

    # Working state (mirror TTSLoop.swift)
    lm_k = voice["lm"]["k"].copy()
    lm_v = voice["lm"]["v"].copy()
    lm_pos = voice["lm"]["len"]

    pos_k = voice["tts_lm"]["k"].copy()
    pos_v = voice["tts_lm"]["v"].copy()
    pos_pos = voice["tts_lm"]["len"]

    neg_k = voice["neg_tts_lm"]["k"].copy()
    neg_v = voice["neg_tts_lm"]["v"].copy()
    neg_pos = voice["neg_tts_lm"]["len"]

    pos_last = voice["tts_lm"]["hidden"][:, -1:, :]  # (1,1,896)
    neg_last = voice["neg_tts_lm"]["hidden"][:, -1:, :]

    # Always have an RNG around — we fall back to it for frames past what the
    # PyTorch reference dumped. For frames [0..args.frames) we still prefer the
    # reference noise so per-step comparisons stay bit-clean.
    rng = SplitMix64(args.seed)

    text_offset = 0
    all_latents: list[np.ndarray] = []
    total_frames = 0
    finished = False
    tts_lm_first_window_hiddens: list[np.ndarray] = []

    while not finished:
        window = tokens[text_offset:text_offset + TEXT_WINDOW]
        text_offset += len(window)

        if window:
            per_token_hidden = []
            for t in window:
                h, lm_k, lm_v = text_decode_step(models["text_decode"], t, lm_k, lm_v, lm_pos)
                per_token_hidden.append(h)
                lm_pos += 1
            for i, t in enumerate(window):
                h, pos_k, pos_v = tts_decode_step(
                    models["tts_decode"], t, per_token_hidden[i],
                    tail_mask=1.0, text_mask=1,
                    k_cache=pos_k, v_cache=pos_v, pos_id=pos_pos)
                pos_last = h
                pos_pos += 1
                if len(tts_lm_first_window_hiddens) < TEXT_WINDOW:
                    tts_lm_first_window_hiddens.append(h[0, -1, :])

        if tts_lm_first_window_hiddens:
            stacked = np.stack(tts_lm_first_window_hiddens, axis=0)
            np.save(dump_dir / "tts_lm_first_window.npy", stacked)

        # Speech window
        for _ in range(SPEECH_WINDOW):
            if total_frames >= args.max_frames:
                finished = True
                break

            should_dump = total_frames < args.frames
            # Save posLast / negLast per frame
            if should_dump:
                np.save(dump_dir / f"posLast_frame{total_frames}.npy",
                        pos_last[0, -1, :].astype(np.float32))
                np.save(dump_dir / f"negLast_frame{total_frames}.npy",
                        neg_last[0, -1, :].astype(np.float32))

            # Initial noise: prefer PyTorch-ref noise where available (so the
            # per-step comparison is isolated from RNG differences), otherwise
            # fall back to the Swift-matching SplitMix64 stream so the frame
            # still runs. We still advance `rng` each frame to keep the Swift
            # parity path deterministic for frames >= args.frames.
            rng_noise = rng.randn(64)
            if args.use_ref_noise:
                noise_path = ref_dir / f"noise_frame{total_frames}.npy"
                if noise_path.exists():
                    init_noise = np.load(noise_path).astype(np.float32)
                else:
                    init_noise = rng_noise
            else:
                init_noise = rng_noise

            latent = denoise_frame(
                models["diffusion"],
                pos_last[0, -1, :], neg_last[0, -1, :],
                schedule, args.steps, args.guidance,
                init_noise, total_frames, dump_dir, should_dump, rng)
            all_latents.append(latent)
            total_frames += 1

            # Acoustic connector
            ac_out = models["acoustic_connector"].predict({
                "latent": latent.reshape(1, 1, 64).astype(np.float32),
            })
            acoustic_embed = ac_out["embedding"]

            # Advance TTS LM branches
            h_pos, pos_k, pos_v = tts_decode_step(
                models["tts_decode"], 1, acoustic_embed,
                tail_mask=1.0, text_mask=0,
                k_cache=pos_k, v_cache=pos_v, pos_id=pos_pos)
            pos_last = h_pos
            pos_pos += 1
            h_neg, neg_k, neg_v = tts_decode_step(
                models["tts_decode"], 1, acoustic_embed,
                tail_mask=1.0, text_mask=0,
                k_cache=neg_k, v_cache=neg_v, pos_id=neg_pos)
            neg_last = h_neg
            neg_pos += 1

            # EOS
            eos_out = models["tts_eos"].predict({
                "hidden": pos_last[:, -1, :].astype(np.float32),
            })
            eos_val = float(np.asarray(eos_out["prob"]).flatten()[0])
            # "prob" output is already sigmoid'd per head conversion
            if eos_val > args.eos_threshold:
                finished = True
                break

        if text_offset >= len(tokens) and not finished:
            continue

    # VAE decode (non-streaming: one shot, T frames at a time, with zero-pad)
    scale = speech_const["speech_scaling_factor"]
    bias = speech_const["speech_bias_factor"]
    audio = np.zeros(0, dtype=np.float32)
    for start in range(0, len(all_latents), VAE_T):
        chunk = all_latents[start:start + VAE_T]
        inp = np.zeros((1, 64, VAE_T), dtype=np.float32)
        for t, lat in enumerate(chunk):
            inp[0, :, t] = (lat / scale) - bias
        vae_out = models["vae"].predict({"latent": inp})
        samples = np.asarray(vae_out["waveform"]).flatten().astype(np.float32)
        valid = len(chunk) * VAE_SAMPLES_PER_FRAME
        audio = np.concatenate([audio, samples[:valid]])

    sf.write(str(dump_dir / "audio.wav"), audio, 24_000)
    print(f"Audio: {len(audio)} samples ({len(audio)/24000:.2f}s)")

    meta = {
        "text": args.text, "voice": args.voice, "seed": args.seed,
        "steps": args.steps, "guidance": args.guidance,
        "n_tokens": len(tokens), "frames_dumped": min(args.frames, total_frames),
        "total_frames": total_frames, "use_ref_noise": args.use_ref_noise,
    }
    (dump_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Trace at: {dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
