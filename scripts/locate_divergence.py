#!/usr/bin/env python3
"""For the 3 non-bit-identical clips, locate WHERE the divergence lives.

Theory: baseline had tail-trim enabled (10 ms fade at cut point), parallel
has tail-trim disabled. The max|diff| should be concentrated in a small
window near the baseline's cut point (the last sample of baseline) and be
~zero everywhere else. If instead the diff is distributed across the whole
waveform, that would indicate concurrency-induced fp drift.
"""
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "out" / "clips"

SUSPECTS = [
    ("b_steps", "steps3_s42"),
    ("b_steps", "steps3_s42_grace"),
    ("b_steps", "steps5_s42_grace"),
]


def main() -> None:
    for setname, clip in SUSPECTS:
        wb = CLIPS / f"_baseline_{setname}" / f"{clip}.wav"
        wp = CLIPS / setname / f"{clip}.wav"
        xb, _ = sf.read(str(wb), dtype="float32")
        xp, _ = sf.read(str(wp), dtype="float32")
        n = min(len(xb), len(xp))
        d = np.abs(xb[:n] - xp[:n])
        # Split into 3 equal regions: head, middle, tail
        third = n // 3
        head_max = d[:third].max()
        mid_max = d[third : 2 * third].max()
        tail_max = d[2 * third :].max()
        # Also: find the argmax and its position as a fraction of the clip
        peak_idx = int(d.argmax())
        frac = peak_idx / n
        # Check last N samples of prefix (tail-trim window suspect)
        last_240 = d[-240:].max()
        print(
            f"{setname}/{clip}: overlap={n} samp, extra={len(xp)-len(xb):+d}\n"
            f"  max|diff|  head={head_max:.2e}  mid={mid_max:.2e}  "
            f"tail={tail_max:.2e}\n"
            f"  argmax at sample {peak_idx} (frac {frac:.3f})  "
            f"last_240samp_max={last_240:.2e}"
        )


if __name__ == "__main__":
    main()
