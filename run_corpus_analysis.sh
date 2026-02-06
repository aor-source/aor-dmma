#!/bin/bash
# Run AoR analysis on full corpus (17 tracks)

OUTPUT_DIR="/Users/alignmentnerd/aor-mir/output/corpus_analysis"
TRACK_LIST="$OUTPUT_DIR/track_list.txt"
HIGH_IMPACT_FILE="$OUTPUT_DIR/high_impact_bars.txt"

echo "=============================================="
echo "AoR CORPUS ANALYSIS - 17 Tracks"
echo "=============================================="
echo ""

# Initialize high impact file
echo "HIGH IMPACT BARS FOR PAPER" > "$HIGH_IMPACT_FILE"
echo "Generated: $(date)" >> "$HIGH_IMPACT_FILE"
echo "==========================================" >> "$HIGH_IMPACT_FILE"
echo "" >> "$HIGH_IMPACT_FILE"

count=0
total=17

while IFS= read -r track; do
    count=$((count + 1))
    filename=$(basename "$track")

    echo ""
    echo "[$count/$total] Processing: $filename"
    echo "----------------------------------------------"

    # Run analysis with CPU flag and visualization
    python3 /Users/alignmentnerd/aor-mir/aor_mir.py \
        "$track" \
        --lexicon /Users/alignmentnerd/aor-mir/aave_lexicon.json \
        --output "$OUTPUT_DIR" \
        --cpu \
        --visualize \
        2>&1 | tee "$OUTPUT_DIR/log_${count}.txt"

    echo ""
done < "$TRACK_LIST"

echo ""
echo "=============================================="
echo "ANALYSIS COMPLETE"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
echo ""
