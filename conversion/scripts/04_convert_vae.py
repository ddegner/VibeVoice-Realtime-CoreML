"""
Phase D — Convert the acoustic tokenizer's decoder (latent → 24 kHz waveform).

Single forward pass, no loop. Watch for ANE buffer limits on the 3200×
upsampling; fall back to GPU-only if needed.

Emits: out/vae_decoder.mlpackage
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, assert_parity, load_manifest, load_vibevoice, resolve  # noqa: E402


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Confirm the exact decoder entry point with module_tree.txt.
        return self.decoder(latent)


def main() -> int:
    import coremltools as ct

    manifest = load_manifest()
    dec_path = manifest["paths"].get("acoustic_decoder") or manifest["paths"].get("acoustic_tok")
    if not dec_path:
        raise SystemExit("acoustic decoder path missing from manifest")

    model = load_vibevoice()
    decoder = resolve(model, dec_path)
    D_latent = manifest["config"]["acoustic_latent_dim"] or 64

    wrapper = DecoderWrapper(decoder).eval()
    # Conv1d layout: (B, C=latent_dim, T). VibeVoice's acoustic_tokenizer
    # downsamples 3200× from 24 kHz audio, so 1 latent frame ≈ 133 ms.
    ex = torch.randn(1, D_latent, 16)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, ex, strict=False)

    mlmodel = ct.convert(
        traced,
        # Fixed T (16) for now — the decoder uses reflection padding that
        # coremltools can't express with dynamic size. Ship multiple fixed-T
        # variants (or bucket) once chunk length is finalized.
        inputs=[ct.TensorType(name="latent", shape=(1, D_latent, 16))],
        outputs=[ct.TensorType(name="waveform")],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    mlmodel.save(str(OUT_DIR / "vae_decoder.mlpackage"))

    with torch.no_grad():
        py = wrapper(ex)
    ml = mlmodel.predict({"latent": ex.numpy()})
    assert_parity(py, ml["waveform"], tol=1e-2, label="vae decoder")  # looser: convolutions accumulate
    return 0


if __name__ == "__main__":
    sys.exit(main())
