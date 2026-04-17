"""
Phase C — Convert the VibeVoice diffusion head to a single-step CoreML model.

The DDPM/DPM-Solver loop and classifier-free-guidance blend live in Swift.
This script only exports `predicted_noise = head(noisy_latent, t, conditioning)`.

Emits: out/diffusion_head.mlpackage
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, assert_parity, load_manifest, load_vibevoice, resolve  # noqa: E402


class DiffusionStepWrapper(torch.nn.Module):
    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head

    def forward(
        self,
        noisy_latent: torch.Tensor,   # (B, T, D_latent)
        timestep: torch.Tensor,       # (B,) int32
        conditioning: torch.Tensor,   # (B, T, D_hidden) — LM hidden states
    ) -> torch.Tensor:
        # The exact call signature here needs to be validated against
        # vibevoice.modular.modular_vibevoice_diffusion_head.forward.
        # Adjust kwargs once 01_inspect_model.py has been run and the head's
        # forward signature is confirmed.
        return self.head(noisy_latent, timestep, conditioning)


def main() -> int:
    import coremltools as ct

    manifest = load_manifest()
    if not manifest["paths"].get("diffusion_head"):
        raise SystemExit("diffusion_head path missing from manifest")

    model = load_vibevoice()
    head = resolve(model, manifest["paths"]["diffusion_head"])
    cfg = manifest["config"]

    # Confirmed from the live model (2026-04-15):
    #   head.config.latent_size   = 64
    #   head.config.hidden_size   = 896
    D_latent = getattr(head.config, "latent_size", 64)
    D_hidden = getattr(head.config, "hidden_size", cfg["hidden_size"])
    B, T = 1, 8  # short chunk; real chunk length is schedule-dependent

    wrapper = DiffusionStepWrapper(head).eval()
    ex_lat = torch.randn(B, T, D_latent)
    # timestep must be float: t_embedder casts its sinusoid back to t.dtype
    # (see modular_vibevoice_diffusion_head.py:88), so int → int Linear error.
    ex_t = torch.tensor([500.0], dtype=torch.float32)
    ex_cond = torch.randn(B, T, D_hidden)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ex_lat, ex_t, ex_cond), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="noisy_latent", shape=(B, ct.RangeDim(1, 4096), D_latent)),
            ct.TensorType(name="timestep", shape=(B,)),
            ct.TensorType(name="conditioning", shape=(B, ct.RangeDim(1, 4096), D_hidden)),
        ],
        outputs=[ct.TensorType(name="predicted_noise")],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,  # fp16 on ANE compounds over 10 steps
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    mlmodel.save(str(OUT_DIR / "diffusion_head.mlpackage"))

    with torch.no_grad():
        py = wrapper(ex_lat, ex_t, ex_cond)
    ml = mlmodel.predict({
        "noisy_latent": ex_lat.numpy(),
        "timestep": ex_t.numpy(),
        "conditioning": ex_cond.numpy(),
    })
    assert_parity(py, ml["predicted_noise"], tol=1e-3, label="diffusion step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
