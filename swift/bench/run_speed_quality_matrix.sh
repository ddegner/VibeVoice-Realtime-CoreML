#!/usr/bin/env bash
# Run speed and listening comparisons for model/compute-unit experiments.
#
# Outputs:
#   bench/listening/<timestamp>/results.jsonl
#   bench/listening/<timestamp>/summary.tsv
#   bench/listening/<timestamp>/*.wav
#
# Optional env:
#   BASE_MODELS_DIR   default ../conversion/out/fp16
#   FP16_MODELS_DIR   optional true compute-precision variant from conversion/scripts/13_prepare_precision_variant.py
#   RUNS              default 2
#   TEXT_FILE         default bench/fixture_2890.txt
#   VOICE             default en-Carter_man
#   SEED              default 42
#   DIFF_STEPS        default 5
#   MAX_FRAMES        default 1500
#   INCLUDE_DIFFUSION_ANE=1 to include a diffusion=ane case
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_MODELS_DIR="${BASE_MODELS_DIR:-../conversion/out/fp16}"
FP16_MODELS_DIR="${FP16_MODELS_DIR:-}"
RUNS="${RUNS:-2}"
TEXT_FILE="${TEXT_FILE:-bench/fixture_2890.txt}"
VOICE="${VOICE:-en-Carter_man}"
SEED="${SEED:-42}"
DIFF_STEPS="${DIFF_STEPS:-5}"
MAX_FRAMES="${MAX_FRAMES:-1500}"
INCLUDE_DIFFUSION_ANE="${INCLUDE_DIFFUSION_ANE:-0}"
OUT_DIR="${OUT_DIR:-bench/listening/$(date +%Y%m%d-%H%M%S)}"
RESULTS="$OUT_DIR/results.jsonl"
SUMMARY="$OUT_DIR/summary.tsv"

if [[ ! -x .build/release/vibevoice-cli ]]; then
    echo "Building release binary..." >&2
    swift build -c release >&2
fi

mkdir -p "$OUT_DIR"
: > "$RESULTS"

TEXT="$(cat "$TEXT_FILE")"

run_case() {
    local label="$1"
    local models_dir="$2"
    local compute_units="${3:-}"

    if [[ ! -d "$models_dir" ]]; then
        echo "Skipping $label: model directory not found: $models_dir" >&2
        return
    fi

    local i=1
    while [[ "$i" -le "$RUNS" ]]; do
        local run_id
        run_id="$(printf "%02d" "$i")"
        local wav="$OUT_DIR/${label}-${run_id}.wav"
        local extra=()
        if [[ -n "$compute_units" ]]; then
            extra+=(--compute-units "$compute_units")
        fi
        if [[ -n "$SEED" ]]; then
            extra+=(--seed "$SEED")
        fi

        echo "[$label] run $run_id/$RUNS -> $wav" >&2
        .build/release/vibevoice-cli "$TEXT" \
            --models "$models_dir" \
            --output "$wav" \
            --voice "$VOICE" \
            --steps "$DIFF_STEPS" \
            --max-frames "$MAX_FRAMES" \
            --json-timing \
            --label "$label" \
            "${extra[@]}" \
            >> "$RESULTS"
        i=$((i + 1))
    done
}

run_case "baseline" "$BASE_MODELS_DIR"
run_case "compute-heads-ane" "$BASE_MODELS_DIR" "connector=ane,eos=ane"
run_case "compute-vae-heads-ane" "$BASE_MODELS_DIR" "vae=ane,connector=ane,eos=ane"

if [[ "$INCLUDE_DIFFUSION_ANE" == "1" ]]; then
    run_case "compute-diffusion-ane" "$BASE_MODELS_DIR" "diffusion=ane"
fi

if [[ -n "$FP16_MODELS_DIR" && -d "$FP16_MODELS_DIR" ]]; then
    run_case "fp16-compute" "$FP16_MODELS_DIR"
    run_case "fp16-compute-heads-ane" "$FP16_MODELS_DIR" "connector=ane,eos=ane"
else
    echo "No FP16_MODELS_DIR supplied; skipping true FP16 compute cases." >&2
fi

python3 - "$RESULTS" "$SUMMARY" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

results = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
groups = defaultdict(list)
for row in rows:
    groups[row.get("label") or "unlabeled"].append(row)

headers = [
    "label", "runs", "total_s_mean", "rtf_mean", "audio_s_mean",
    "tts_lm_s", "diffusion_s", "vae_s", "peak", "rms",
    "silent_run_s", "q4_over_q1", "glitches", "first_wav",
]

def mean(values):
    return statistics.mean(values) if values else 0

lines = ["\t".join(headers)]
for label in sorted(groups):
    items = groups[label]
    timing = [item["timing"] for item in items]
    stats = [item["audio_stats"] for item in items]
    first_wav = next((item.get("output") for item in items if item.get("output")), "")
    values = [
        label,
        str(len(items)),
        f"{mean([t['total'] for t in timing]):.2f}",
        f"{mean([item['rt_factor'] for item in items]):.3f}",
        f"{mean([item['audio_duration_sec'] for item in items]):.2f}",
        f"{mean([t['tts_lm_decode'] for t in timing]):.2f}",
        f"{mean([t['diffusion'] for t in timing]):.2f}",
        f"{mean([t['vae_decode'] for t in timing]):.2f}",
        f"{mean([s['peak'] for s in stats]):.4f}",
        f"{mean([s['rms'] for s in stats]):.4f}",
        f"{mean([s['longest_silent_run_s'] for s in stats]):.3f}",
        f"{mean([s['q4_over_q1'] for s in stats]):.3f}",
        f"{mean([s['glitch_windows'] for s in stats]):.1f}",
        first_wav,
    ]
    lines.append("\t".join(values))

summary.write_text("\n".join(lines) + "\n")
print(summary.read_text())
PY

echo "WAVs and timing written to: $OUT_DIR"
