#!/usr/bin/env python3
"""Compare parallel vs sequential clip matrix.

For each clip name present in both _report.md (parallel) and
_report_baseline_sequential.md (sequential):
  * RTF delta (lower = faster)
  * wall delta
  * sample-count delta (should be 0 if the EOS decisions agreed)
  * if same length: max|x-y| on raw PCM (bit-level parity)
  * if different length: truncate to min length and compare — flag if the
    prefix disagrees substantially (would indicate a concurrency-induced
    numeric divergence rather than just tail-trim disable)

Also recomputes per-clip peak / rms / spec_flux for both and shows the diff
on the common prefix.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "out" / "clips"
PAR_REPORT = CLIPS / "_report.md"
BASE_REPORT = CLIPS / "_report_baseline_sequential.md"


def parse_report(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    txt = path.read_text().splitlines()
    for line in txt:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 10:
            continue
        if parts[0] in ("set", ":---") or parts[0].startswith("---"):
            continue
        try:
            rows[(parts[0], parts[1])] = {
                "voice": parts[2],
                "chars": int(parts[3]),
                "steps": int(parts[4]),
                "seed": int(parts[5]),
                "cfg": float(parts[6]),
                "samples": int(parts[7]),
                "wall": float(parts[8]),
                "rtf": float(parts[9]),
            }
        except ValueError:
            continue
    return rows


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def spec_flux(x: np.ndarray, sr: int = 24000, win: int = 1024, hop: int = 256) -> float:
    if len(x) < win * 2:
        return 0.0
    frames = []
    for i in range(0, len(x) - win, hop):
        seg = x[i : i + win]
        mag = np.abs(np.fft.rfft(seg * np.hanning(win)))
        frames.append(mag)
    frames = np.stack(frames)  # (F, K)
    d = np.diff(frames, axis=0)
    d = np.clip(d, 0, None)  # positive-only (half-wave rectified)
    return float(d.sum() / max(1, d.size))


def clip_paths(set_name: str, clip: str, variant: str) -> Path:
    root = CLIPS / (f"_baseline_{set_name}" if variant == "baseline" else set_name)
    return root / f"{clip}.wav"


def main() -> None:
    par = parse_report(PAR_REPORT)
    base = parse_report(BASE_REPORT)
    keys = sorted(set(par.keys()) & set(base.keys()))

    print(f"Comparing {len(keys)} clips (parallel vs sequential)\n")

    print(
        "| set | clip | samples Δ | RTF base | RTF par | RTF Δ% "
        "| max|diff| on prefix | peak Δ | rms Δ | flux Δ |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    speedups = []
    bit_identical = 0
    diverged = 0

    for k in keys:
        sset, cname = k
        p, b = par[k], base[k]
        wb = clip_paths(sset, cname, "baseline")
        wp = clip_paths(sset, cname, "parallel")
        if not wb.exists() or not wp.exists():
            print(f"| {sset} | {cname} | MISSING_WAV | — | — | — | — | — | — | — |")
            continue
        xb = read_wav(wb)
        xp = read_wav(wp)

        n = min(len(xb), len(xp))
        diff = np.abs(xb[:n] - xp[:n]).max()
        exact = diff == 0.0 and len(xb) == len(xp)
        if exact:
            bit_identical += 1
        elif diff > 0.02:
            diverged += 1

        # RMS/peak/flux on common prefix
        peak_b = float(np.abs(xb[:n]).max())
        peak_p = float(np.abs(xp[:n]).max())
        rms_b = float(np.sqrt((xb[:n] ** 2).mean()))
        rms_p = float(np.sqrt((xp[:n] ** 2).mean()))
        f_b = spec_flux(xb[:n])
        f_p = spec_flux(xp[:n])

        rtf_pct = (p["rtf"] - b["rtf"]) / b["rtf"] * 100.0
        speedups.append(rtf_pct)
        ds = p["samples"] - b["samples"]

        print(
            f"| {sset} | {cname} | {ds:+d} | {b['rtf']:.2f} | {p['rtf']:.2f} | "
            f"{rtf_pct:+.1f}% | {diff:.2e} | {peak_p - peak_b:+.4f} | "
            f"{rms_p - rms_b:+.4f} | {f_p - f_b:+.2e} |"
        )

    if speedups:
        avg = sum(speedups) / len(speedups)
        best = min(speedups)
        worst = max(speedups)
        print(f"\nRTF change: mean {avg:+.1f}%  best {best:+.1f}%  worst {worst:+.1f}%")
    print(
        f"Bit-identical clips: {bit_identical}/{len(keys)}  "
        f"diverged (>0.02 max|diff|) on prefix: {diverged}"
    )


if __name__ == "__main__":
    main()
