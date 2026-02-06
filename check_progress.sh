#!/bin/bash
# AoR-DMMA Quick Progress Check
# Usage: bash check_progress.sh [output_dir] [tracks_file]

OUTPUT_DIR="${1:-output/full_v71}"
TRACKS_FILE="${2:-tracks.txt}"

cd /Users/alignmentnerd/aor-dmma

TOTAL=$(grep -v '^#' "$TRACKS_FILE" | grep -v '^$' | wc -l | tr -d ' ')
COMPLETED=$(ls "$OUTPUT_DIR"/*_sovereign.json 2>/dev/null | wc -l | tr -d ' ')
PCT=$((COMPLETED * 100 / TOTAL))

# Bar
BAR_W=30
FILLED=$((PCT * BAR_W / 100))
EMPTY=$((BAR_W - FILLED))
BAR=""
for ((i=0; i<FILLED; i++)); do BAR="${BAR}█"; done
for ((i=0; i<EMPTY; i++)); do BAR="${BAR}░"; done

# Last completed
LAST=$(ls -t "$OUTPUT_DIR"/*_sovereign.json 2>/dev/null | head -1)
[ -n "$LAST" ] && LAST_NAME=$(basename "$LAST" _sovereign.json) || LAST_NAME="(none yet)"

echo ""
echo "  AoR v7.1 MLX - Progress"
echo "  ────────────────────────────────────────"
echo "  ${BAR}  ${PCT}%"
echo ""
echo "  Completed:  ${COMPLETED} / ${TOTAL}"
echo "  Last done:  ${LAST_NAME}"
echo ""

# List completed tracks
if [ "$COMPLETED" -gt 0 ]; then
    echo "  Finished:"
    ls -t "$OUTPUT_DIR"/*_sovereign.json 2>/dev/null | while read f; do
        echo "    ✅ $(basename "$f" _sovereign.json)"
    done
fi

echo ""
