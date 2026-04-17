"""
Phase B/C/D gate — end-to-end numerical check: run a fixed prompt through
PyTorch and through the 4 .mlpackage artifacts, compare waveforms.

Usage: python scripts/05_parity_check.py --text "The quick brown fox."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import OUT_DIR, load_vibevoice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--tol-rmse", type=float, default=5e-3)
    args = ap.parse_args()

    # Reference: full PyTorch pipeline.
    model = load_vibevoice()
    # TODO(phase E wiring): call model.generate(...) with the same seed +
    # scheduler config as the Swift pipeline, save reference.wav.

    # CoreML: load all four .mlpackages, run the Swift-equivalent pipeline in
    # Python (numpy schedulers, numpy CFG), save coreml.wav.

    # Compare:
    #   - RMSE on normalised waveforms
    #   - STFT magnitude spectrogram MSE (more robust to phase drift)
    raise SystemExit(
        "05_parity_check is scaffolded but not implemented — fill in once "
        "Phases B/C/D produce working .mlpackages."
    )


if __name__ == "__main__":
    sys.exit(main())
