#!/usr/bin/env bash
# Run the VibeVoice benchmark fixture N times and emit one JSON line per run
# to bench/results/<git-sha>-<label>.jsonl. Defaults: N=3, label="adhoc".
#
# Usage:
#   bench/run.sh [LABEL] [N]
#
# Examples:
#   bench/run.sh baseline      # 3 runs, label "baseline"
#   bench/run.sh phase1 5      # 5 runs, label "phase1"
#
# Reads:
#   bench/fixture_2890.txt   — canonical 2890-char text (Pride and Prejudice)
# Env overrides:
#   MODELS_DIR  — default ../conversion/out/fp16 relative to swift/
#   VOICE       — default en-Carter_man
#   SEED        — default 42
#   MAX_FRAMES  — default 1500 (~200s cap)
#   COMPUTE_UNITS — passed through to --compute-units (e.g. "vae=ane,connector=ane")
#
# The script always discards run #1 if N>=2 (cold cache / ANE first-compile
# noise), so set N=4 to get 3 measured + 1 warmup discarded.
set -euo pipefail

LABEL="${1:-adhoc}"
N="${2:-3}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${MODELS_DIR:-../conversion/out/fp16}"
VOICE="${VOICE:-en-Carter_man}"
SEED="${SEED:-42}"
MAX_FRAMES="${MAX_FRAMES:-1500}"

if [[ ! -x .build/release/vibevoice-cli ]]; then
    echo "Building release binary…" >&2
    swift build -c release >&2
fi

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    DIRTY="-dirty"
fi
export GIT_SHA="${GIT_SHA}${DIRTY}"

OUT="bench/results/${GIT_SHA}-${LABEL}.jsonl"
mkdir -p bench/results
: > "$OUT"

FIXTURE="$(cat bench/fixture_2890.txt)"
TMPWAV="$(mktemp -t vibevoice-bench.XXXXXX.wav)"
trap 'rm -f "$TMPWAV"' EXIT

echo "Running $N iteration(s): label=$LABEL sha=$GIT_SHA" >&2
echo "  models=$MODELS_DIR voice=$VOICE seed=$SEED maxFrames=$MAX_FRAMES" >&2
echo "  → $OUT" >&2

EXTRA=()
if [[ -n "${COMPUTE_UNITS:-}" ]]; then
    EXTRA+=(--compute-units "$COMPUTE_UNITS")
    echo "  compute-units=$COMPUTE_UNITS" >&2
fi
if [[ -n "${DIFF_STEPS:-}" ]]; then
    EXTRA+=(--steps "$DIFF_STEPS")
    echo "  diffusion-steps=$DIFF_STEPS" >&2
fi
if [[ -n "${TTS_HANDLES:-}" ]]; then
    EXTRA+=(--tts-handles "$TTS_HANDLES")
    echo "  tts-handles=$TTS_HANDLES" >&2
fi

for i in $(seq 1 "$N"); do
    echo "  iter $i/$N…" >&2
    .build/release/vibevoice-cli "$FIXTURE" \
        --models "$MODELS_DIR" \
        --output "$TMPWAV" \
        --voice "$VOICE" \
        --seed "$SEED" \
        --max-frames "$MAX_FRAMES" \
        --json-timing \
        --label "$LABEL" \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        >> "$OUT"
done

echo "done. $(wc -l < "$OUT") lines written." >&2
echo "$OUT"
