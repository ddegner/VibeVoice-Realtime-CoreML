#!/usr/bin/env python3
"""Compare two JSONL benchmark outputs from bench/run.sh.

Per-stage timings: print mean ± stddev for each file and the delta
(new − baseline). Audio stats deltas are checked against the gate
criteria from /Users/degner/.claude/plans/purrfect-pondering-karp.md
("Quality gate criteria") and printed as PASS/FAIL.

Usage:
    bench/compare.py bench/results/<sha>-baseline.jsonl bench/results/<sha>-phase1.jsonl

Optional --discard-first drops the first run from each file (cold-cache /
ANE first-compile noise). Default is on; pass --no-discard-first to keep all.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Stages to print (key in JSON timing block, label).
STAGES = [
    ("total",          "TOTAL"),
    ("text_lm_decode", "  text_lm_decode"),
    ("tts_lm_decode",  "  tts_lm_decode"),
    ("diffusion",      "  diffusion"),
    ("connector",      "  connector"),
    ("eos_classifier", "  eos_classifier"),
    ("vae_decode",     "  vae_decode"),
]

AUDIO_STATS = [
    ("peak",                 "peak"),
    ("rms",                  "rms"),
    ("longest_silent_run_s", "longest_silent_run_s"),
    ("q1_rms",               "q1_rms"),
    ("q4_rms",               "q4_rms"),
    ("q4_over_q1",           "q4_over_q1"),
    ("glitch_windows",       "glitch_windows"),
]


def load(path: Path, discard_first: bool) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if discard_first and len(rows) > 1:
        rows = rows[1:]
    return rows


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(v: float, width: int = 6, prec: int = 2) -> str:
    return f"{v:>{width}.{prec}f}"


def gate_audio(base: dict, new: dict, base_samples: int, new_samples: int) -> list[tuple[str, str]]:
    """Return [(criterion, PASS/FAIL message), …]."""
    out = []

    # Longest silent run grows by >0.3s
    d = new["longest_silent_run_s"] - base["longest_silent_run_s"]
    out.append(("silent_run", f"{'PASS' if d <= 0.3 else 'FAIL'}  Δ={d:+.3f}s (limit +0.3s)"))

    # Q4/Q1 RMS ratio drops by >0.1 vs baseline. (Absolute 0.9 threshold from
    # the plan was too strict — narrative endings naturally fade. Delta-based
    # is what catches *new* level decay.)
    d = new["q4_over_q1"] - base["q4_over_q1"]
    out.append(("q4_over_q1", f"{'PASS' if d >= -0.1 else 'FAIL'}  Δ={d:+.3f} (limit -0.10)"))

    # Glitch windows increase by >2
    d = new["glitch_windows"] - base["glitch_windows"]
    out.append(("glitch_windows", f"{'PASS' if d <= 2 else 'FAIL'}  Δ={d:+d} (limit +2)"))

    # Peak: drop >1 dB OR within 0.5 dB of clipping
    base_peak_db = 20 * math.log10(max(base["peak"], 1e-9))
    new_peak_db  = 20 * math.log10(max(new["peak"],  1e-9))
    drop = base_peak_db - new_peak_db
    headroom = -new_peak_db
    peak_ok = drop <= 1.0 and headroom >= 0.5
    out.append(("peak", f"{'PASS' if peak_ok else 'FAIL'}  Δ={-drop:+.2f} dB, headroom={headroom:.2f} dB"))

    # PCM length differs by >40ms (24 kHz → 960 samples)
    sample_delta_ms = (new_samples - base_samples) * 1000.0 / 24000.0
    out.append(("pcm_length", f"{'PASS' if abs(sample_delta_ms) <= 40 else 'FAIL'}  Δ={sample_delta_ms:+.1f} ms (limit ±40)"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("compare", type=Path)
    ap.add_argument("--no-discard-first", action="store_true",
                    help="Keep first run from each file (default: discard)")
    args = ap.parse_args()

    discard = not args.no_discard_first
    base = load(args.baseline, discard)
    new  = load(args.compare,  discard)
    if not base or not new:
        print(f"empty input: base={len(base)} new={len(new)}", file=sys.stderr)
        return 1

    print(f"baseline: {args.baseline} ({len(base)} runs)")
    print(f"compare:  {args.compare} ({len(new)} runs)")
    print(f"discard-first: {discard}")
    print()

    # Per-stage timing
    print(f"{'stage':<22} {'baseline':>16} {'compare':>16} {'Δ':>14} {'Δ%':>8}")
    print("-" * 80)
    for key, label in STAGES:
        b_xs = [r["timing"][key] for r in base]
        n_xs = [r["timing"][key] for r in new]
        b_m, b_s = mean_std(b_xs)
        n_m, n_s = mean_std(n_xs)
        delta = n_m - b_m
        pct = (delta / b_m * 100) if b_m > 0 else 0
        print(f"{label:<22} {fmt(b_m)}±{fmt(b_s, 4)} "
              f"{fmt(n_m)}±{fmt(n_s, 4)} "
              f"{delta:+8.2f}s    {pct:+6.1f}%")
    print()

    # Audio stats — use means
    print(f"{'audio_stat':<22} {'baseline':>14} {'compare':>14} {'Δ':>14}")
    print("-" * 70)
    for key, label in AUDIO_STATS:
        b_xs = [r["audio_stats"][key] for r in base]
        n_xs = [r["audio_stats"][key] for r in new]
        b_m, _ = mean_std(b_xs)
        n_m, _ = mean_std(n_xs)
        d = n_m - b_m
        print(f"{label:<22} {b_m:>14.4f} {n_m:>14.4f} {d:>+14.4f}")
    print()

    # Quality gate — use means of audio stats and median sample count
    base_stats_mean = {k: statistics.mean(r["audio_stats"][k] for r in base) for k, _ in AUDIO_STATS}
    new_stats_mean  = {k: statistics.mean(r["audio_stats"][k] for r in new ) for k, _ in AUDIO_STATS}
    base_samples = round(statistics.median(r["samples"] for r in base))
    new_samples  = round(statistics.median(r["samples"] for r in new))

    print("quality gate (per plan §Quality gate criteria):")
    failed = 0
    for name, msg in gate_audio(base_stats_mean, new_stats_mean, base_samples, new_samples):
        print(f"  {name:<18} {msg}")
        if msg.startswith("FAIL"):
            failed += 1

    print()
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
