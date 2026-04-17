"""
Generate reference audio from the PyTorch VibeVoice model for aural comparison
against the Swift CLI CoreML output.

Uses the upstream demo's exact call path: loads a cached voice prompt (.pt),
runs `process_input_with_cached_prompt`, and calls `model.generate`.

Emits:
  out/ref_pytorch.wav

Usage:
  python scripts/07_reference_audio.py "Hello world, this is a test."
  python scripts/07_reference_audio.py   # uses default prompt
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1] / "third_party" / "VibeVoice" / "demo"
VOICE_PT = DEMO_DIR / "voices" / "streaming_model" / "en-Frank_man.pt"


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello world, this is a test."
    print(f"Prompt: {prompt!r}")

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    print("Loading model…")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)

    processor = VibeVoiceStreamingProcessor.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B"
    )

    print(f"Loading voice: {VOICE_PT.name}")
    cached_prompt = torch.load(str(VOICE_PT), map_location="cpu", weights_only=False)

    inputs = processor.process_input_with_cached_prompt(
        text=prompt,
        cached_prompt=cached_prompt,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cpu")
    print(f"  input_ids {inputs['input_ids'].shape}  tts_text_ids {inputs['tts_text_ids'].shape}")

    print("Running generate (cfg_scale=3.0, 20 ddpm steps, CPU — may take minutes)…")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=3.0,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        all_prefilled_outputs=copy.deepcopy(cached_prompt),
    )

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        wav = outputs.speech_outputs[0].squeeze().cpu().numpy().astype(np.float32)
        print(
            f"Audio: {len(wav)} samples, {len(wav)/24000:.2f}s, "
            f"range [{wav.min():.4f}, {wav.max():.4f}], "
            f"rms={np.sqrt((wav**2).mean()):.4f}"
        )
        out_path = OUT_DIR / "ref_pytorch.wav"
        sf.write(str(out_path), np.clip(wav, -1, 1), 24000, subtype="FLOAT")
        print(f"Wrote {out_path}")
    else:
        print("ERROR: no speech output generated")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
