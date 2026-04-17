#!/bin/bash
# Generate a matrix of test clips for quality + speed evaluation.
#
# Output: out/clips/<set>/<name>.wav and out/clips/_report.md
# Each group is designed to stress a specific dimension:
#   A: voice variety (same text, 3 voices)
#   B: step count (speed-quality tradeoff: 3, 5, 10 steps)
#   C: text length (short, medium, long)
#   D: seed sweep (stutter regression check across 7 seeds)
#
# Run from the project root:
#   bash scripts/generate_test_clips.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$ROOT/swift/.build/debug/vibevoice-cli"
MODELS="$ROOT/conversion/out"
OUT="$ROOT/out/clips"
REPORT="$OUT/_report.md"

if [ ! -x "$CLI" ]; then
    echo "CLI not built at $CLI — running 'swift build' in swift/"
    (cd "$ROOT/swift" && swift build -c debug)
fi

mkdir -p "$OUT/a_voices" "$OUT/b_steps" "$OUT/c_length" "$OUT/d_seeds"

# Paragraph for length test
LONG_TEXT="The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. How much wood would a woodchuck chuck if a woodchuck could chuck wood? Peter Piper picked a peck of pickled peppers."

SHORT_TEXT="Hello, world. This is a quick test."
FOX="The quick brown fox jumps over the lazy dog."

echo "# VibeVoice CoreML test clip report" > "$REPORT"
echo "Generated $(date)" >> "$REPORT"
echo "" >> "$REPORT"
echo "| set | name | voice | text_chars | steps | seed | cfg | samples | wall (s) | RTF |" >> "$REPORT"
echo "|-----|------|-------|-----------:|------:|-----:|----:|--------:|---------:|----:|" >> "$REPORT"

run_one() {
    local setdir="$1" name="$2" voice="$3" steps="$4" seed="$5" cfg="$6" text="$7"
    local out="$OUT/$setdir/$name.wav"
    echo ""
    echo "=== $setdir/$name  voice=$voice steps=$steps seed=$seed cfg=$cfg ==="
    # Capture stdout to pull timing + samples
    local log
    log=$("$CLI" "$text" --models "$MODELS" --voice "$voice" --steps "$steps" \
          --guidance "$cfg" --seed "$seed" --output "$out" 2>&1 \
          | grep -vE "^E5RT" || true)
    echo "$log"
    local samples wall rtf
    samples=$(echo "$log" | grep -E "^Generated " | awk '{print $2}')
    wall=$(echo "$log" | grep -E "Real-time factor" | awk -F"[()]" '{print $2}' | awk '{print $1}' | sed 's/s//')
    rtf=$(echo "$log" | grep -E "Real-time factor" | awk '{print $3}' | sed 's/x//')
    local chars=${#text}
    echo "| $setdir | $name | $voice | $chars | $steps | $seed | $cfg | $samples | $wall | $rtf |" >> "$REPORT"
}

# --------------------------------------------------------------------
# Set A: Voice variety
# --------------------------------------------------------------------
echo ""
echo "========== SET A: voice variety =========="
run_one a_voices "frank_male"    en-Frank_man   5 42 1.5 "$FOX"
run_one a_voices "grace_female"  en-Grace_woman 5 42 1.5 "$FOX"
run_one a_voices "mike_male"     en-Mike_man    5 42 1.5 "$FOX"
run_one a_voices "emma_female"   en-Emma_woman  5 42 1.5 "$FOX"
run_one a_voices "carter_male"   en-Carter_man  5 42 1.5 "$FOX"
run_one a_voices "davis_male"    en-Davis_man   5 42 1.5 "$FOX"

# --------------------------------------------------------------------
# Set B: Step count (speed vs quality)
# --------------------------------------------------------------------
echo ""
echo "========== SET B: step count sweep =========="
for steps in 3 5 10 20; do
    run_one b_steps "steps${steps}_s42" en-Frank_man $steps 42 1.5 "$FOX"
done
# Also try a woman voice to see if step count interacts differently
for steps in 3 5 10; do
    run_one b_steps "steps${steps}_s42_grace" en-Grace_woman $steps 42 1.5 "$FOX"
done

# --------------------------------------------------------------------
# Set C: Text length
# --------------------------------------------------------------------
echo ""
echo "========== SET C: text length =========="
run_one c_length "short"   en-Frank_man 5 42 1.5 "$SHORT_TEXT"
run_one c_length "medium"  en-Frank_man 5 42 1.5 "$FOX"
run_one c_length "long"    en-Frank_man 5 42 1.5 "$LONG_TEXT"

# --------------------------------------------------------------------
# Set D: Seed sweep (stutter regression check)
# --------------------------------------------------------------------
echo ""
echo "========== SET D: seed sweep =========="
for seed in 1 7 42 123 999 2024 31337; do
    run_one d_seeds "seed_${seed}" en-Frank_man 5 $seed 1.5 "$FOX"
done

echo ""
echo "=========="
echo "Done. Clips in $OUT, report at $REPORT"
