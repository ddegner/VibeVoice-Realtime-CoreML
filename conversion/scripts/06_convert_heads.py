"""
Phase E — export the small heads the streaming inference loop needs:

1. `acoustic_connector` (SpeechConnector): (B, T, 64) → (B, T, 896).
   Injected back into the TTS LM as `lm_last_hidden_state` for each speech
   step (see modeling_vibevoice_streaming_inference.py:804).

2. `tts_eos_classifier` (BinaryClassifier): (B, 896) → (B, 1) logit.
   Run on the TTS LM's last hidden state each speech step; sigmoid > 0.5
   stops generation for that sample (line 848).

Also dumps the two scalar buffers (`speech_scaling_factor`,
`speech_bias_factor`) into `out/speech_constants.json` so the Swift side can
apply the `speech_latent / scale - bias` normalization before calling the
VAE decoder (line 783).

Emits:
  out/acoustic_connector.mlpackage
  out/tts_eos_classifier.mlpackage
  out/speech_constants.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, assert_parity, load_vibevoice  # noqa: E402


class ConnectorWrapper(torch.nn.Module):
    def __init__(self, connector: torch.nn.Module):
        super().__init__()
        self.connector = connector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.connector(latent)


class EOSWrapper(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(hidden))


def main() -> int:
    import coremltools as ct

    model = load_vibevoice()
    connector = model.model.acoustic_connector.eval()
    classifier = model.tts_eos_classifier.eval()

    scale = float(model.model.speech_scaling_factor.detach().cpu())
    bias = float(model.model.speech_bias_factor.detach().cpu())
    consts = {
        "speech_scaling_factor": scale,
        "speech_bias_factor": bias,
        "note": "Apply as: normalized = raw_latent / scale - bias before VAE decode.",
    }
    with (OUT_DIR / "speech_constants.json").open("w") as f:
        json.dump(consts, f, indent=2)
    print(f"speech_scaling_factor={scale:.6f} speech_bias_factor={bias:.6f}")

    # --- acoustic_connector ---------------------------------------------------
    cw = ConnectorWrapper(connector).eval()
    ex_lat = torch.randn(1, 1, 64)
    with torch.no_grad():
        traced = torch.jit.trace(cw, ex_lat, strict=False)
    ml_connector = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=(1, ct.RangeDim(1, 4096), 64))],
        outputs=[ct.TensorType(name="embedding")],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_connector.save(str(OUT_DIR / "acoustic_connector.mlpackage"))
    with torch.no_grad():
        py = cw(ex_lat)
    ml = ml_connector.predict({"latent": ex_lat.numpy()})
    assert_parity(py, ml["embedding"], tol=1e-5, label="acoustic_connector")

    # --- tts_eos_classifier ---------------------------------------------------
    ew = EOSWrapper(classifier).eval()
    ex_hid = torch.randn(1, 896)
    with torch.no_grad():
        traced = torch.jit.trace(ew, ex_hid, strict=False)
    ml_eos = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden", shape=(1, 896))],
        outputs=[ct.TensorType(name="prob")],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml_eos.save(str(OUT_DIR / "tts_eos_classifier.mlpackage"))
    with torch.no_grad():
        py = ew(ex_hid)
    ml = ml_eos.predict({"hidden": ex_hid.numpy()})
    assert_parity(py, ml["prob"], tol=1e-5, label="tts_eos_classifier")

    return 0


if __name__ == "__main__":
    sys.exit(main())
