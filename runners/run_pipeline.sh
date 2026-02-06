#!/bin/bash
#
# AoR-MIR v7.0 - MODULAR PIPELINE RUNNER
# =======================================
# Orchestrates component execution with isolated venvs
#
# Usage: ./run_pipeline.sh <audio_file_or_dir> [output_dir] [lexicon.json]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPONENTS_DIR="$PROJECT_DIR/components"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     AoR-MIR v7.0 - MAXIMUM INTELLIGENCE RESONANCE       ║"
echo "║          Modular Pipeline for Apple Silicon M5           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Arguments
INPUT="$1"
OUTPUT_DIR="${2:-$PROJECT_DIR/output}"
LEXICON="$3"

if [ -z "$INPUT" ]; then
    echo -e "${RED}Usage: $0 <audio_file_or_dir> [output_dir] [lexicon.json]${NC}"
    exit 1
fi

# Create work directory
WORK_DIR="$OUTPUT_DIR/.work"
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

# Detect venvs or use system Python
WHISPER_PYTHON="${PROJECT_DIR}/venv-whisper/bin/python3"
LIBROSA_PYTHON="${PROJECT_DIR}/venv-librosa/bin/python3"
SYSTEM_PYTHON="python3"

# Fallback to system or project venv
if [ ! -f "$WHISPER_PYTHON" ]; then
    if [ -f "$PROJECT_DIR/venv/bin/python3" ]; then
        WHISPER_PYTHON="$PROJECT_DIR/venv/bin/python3"
    else
        WHISPER_PYTHON="$SYSTEM_PYTHON"
    fi
fi

if [ ! -f "$LIBROSA_PYTHON" ]; then
    if [ -f "$PROJECT_DIR/venv/bin/python3" ]; then
        LIBROSA_PYTHON="$PROJECT_DIR/venv/bin/python3"
    else
        LIBROSA_PYTHON="$SYSTEM_PYTHON"
    fi
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT_DIR"
echo "  Work Dir:   $WORK_DIR"
echo "  Lexicon:    ${LEXICON:-DEFAULT}"
echo ""

# Collect audio files
if [ -d "$INPUT" ]; then
    AUDIO_FILES=$(find "$INPUT" -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" -o -name "*.m4a" \) | sort)
else
    AUDIO_FILES="$INPUT"
fi

FILE_COUNT=$(echo "$AUDIO_FILES" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $FILE_COUNT audio file(s)${NC}"
echo ""

# Process each file
PROCESSED=0
FAILED=0

for AUDIO_FILE in $AUDIO_FILES; do
    [ -z "$AUDIO_FILE" ] && continue

    FILENAME=$(basename "$AUDIO_FILE")
    STEM="${FILENAME%.*}"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Processing: $FILENAME${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Component 1: Transcription
    echo -e "\n${GREEN}[1/4] Transcription...${NC}"
    if $WHISPER_PYTHON "$COMPONENTS_DIR/01_transcribe.py" "$AUDIO_FILE" "$WORK_DIR" 2>&1; then
        echo -e "${GREEN}      ✓ Transcription complete${NC}"
    else
        echo -e "${RED}      ✗ Transcription failed${NC}"
        ((FAILED++))
        continue
    fi

    # Component 2: Feature Extraction
    echo -e "\n${GREEN}[2/4] Feature Extraction...${NC}"
    if $LIBROSA_PYTHON "$COMPONENTS_DIR/02_features.py" "$AUDIO_FILE" "$WORK_DIR" 2>&1; then
        echo -e "${GREEN}      ✓ Features extracted${NC}"
    else
        echo -e "${RED}      ✗ Feature extraction failed${NC}"
        ((FAILED++))
        continue
    fi

    # Component 3: AAVE Analysis
    echo -e "\n${GREEN}[3/4] AAVE Analysis...${NC}"
    TRANSCRIPT_JSON="$WORK_DIR/${STEM}_transcript.json"
    if [ -n "$LEXICON" ]; then
        $SYSTEM_PYTHON "$COMPONENTS_DIR/03_aave.py" "$TRANSCRIPT_JSON" "$WORK_DIR" "$LEXICON" 2>&1
    else
        $SYSTEM_PYTHON "$COMPONENTS_DIR/03_aave.py" "$TRANSCRIPT_JSON" "$WORK_DIR" 2>&1
    fi
    echo -e "${GREEN}      ✓ AAVE analysis complete${NC}"

    # Component 4: Final Assembly
    echo -e "\n${GREEN}[4/4] Final Assembly...${NC}"
    $SYSTEM_PYTHON "$COMPONENTS_DIR/04_assemble.py" "$STEM" "$WORK_DIR" "$OUTPUT_DIR" 2>&1
    echo -e "${GREEN}      ✓ Assembly complete${NC}"

    ((PROCESSED++))
    echo ""
done

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗"
echo -e "║                    PIPELINE COMPLETE                      ║"
echo -e "╠══════════════════════════════════════════════════════════╣"
echo -e "║  Processed: $PROCESSED                                            ║"
echo -e "║  Failed:    $FAILED                                            ║"
echo -e "║  Output:    $OUTPUT_DIR"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}"

# List outputs
echo -e "\n${YELLOW}Output Files:${NC}"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files yet)"
