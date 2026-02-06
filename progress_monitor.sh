#!/bin/bash
# AoR-DMMA Progress Monitor
# Run in a separate Kitty tab to watch analysis progress

OUTPUT_DIR="${1:-output}"
TRACKS_FILE="${2:-tracks.txt}"
TOTAL=$(grep -v '^#' "$TRACKS_FILE" | grep -v '^$' | wc -l | tr -d ' ')
START_TIME=$(date +%s)
BAR_WIDTH=40

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
DIM='\033[0;90m'
RESET='\033[0m'
BOLD='\033[1m'

clear
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║${WHITE}${BOLD}     AoR v7.1 MLX - PROGRESS MONITOR                     ${CYAN}║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${DIM}Output dir: ${OUTPUT_DIR}${RESET}"
echo -e "${DIM}Total tracks: ${TOTAL}${RESET}"
echo ""

while true; do
    COMPLETED=$(ls "$OUTPUT_DIR"/*_sovereign.json 2>/dev/null | wc -l | tr -d ' ')
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))

    # Calculate percentage
    if [ "$TOTAL" -gt 0 ]; then
        PCT=$((COMPLETED * 100 / TOTAL))
    else
        PCT=0
    fi

    # Calculate ETA
    if [ "$COMPLETED" -gt 0 ] && [ "$COMPLETED" -lt "$TOTAL" ]; then
        SECS_PER_TRACK=$((ELAPSED / COMPLETED))
        REMAINING=$(( (TOTAL - COMPLETED) * SECS_PER_TRACK ))
        ETA_MIN=$((REMAINING / 60))
        ETA_SEC=$((REMAINING % 60))
        ETA_STR="${ETA_MIN}m ${ETA_SEC}s"
    elif [ "$COMPLETED" -ge "$TOTAL" ]; then
        ETA_STR="DONE"
    else
        ETA_STR="calculating..."
    fi

    # Elapsed time formatted
    E_MIN=$((ELAPSED / 60))
    E_SEC=$((ELAPSED % 60))
    ELAPSED_STR="${E_MIN}m ${E_SEC}s"

    # Build progress bar
    if [ "$TOTAL" -gt 0 ]; then
        FILLED=$((PCT * BAR_WIDTH / 100))
    else
        FILLED=0
    fi
    EMPTY=$((BAR_WIDTH - FILLED))

    BAR=""
    for ((i=0; i<FILLED; i++)); do BAR="${BAR}█"; done
    for ((i=0; i<EMPTY; i++)); do BAR="${BAR}░"; done

    # Get last completed track
    LAST_FILE=$(ls -t "$OUTPUT_DIR"/*_sovereign.json 2>/dev/null | head -1)
    if [ -n "$LAST_FILE" ]; then
        LAST_TRACK=$(basename "$LAST_FILE" _sovereign.json)
    else
        LAST_TRACK="(waiting...)"
    fi

    # Move cursor up and redraw
    tput cup 6 0

    if [ "$COMPLETED" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        COLOR=$GREEN
    else
        COLOR=$YELLOW
    fi

    echo -e "  ${COLOR}${BAR}${RESET}  ${BOLD}${PCT}%${RESET}    "
    echo ""
    echo -e "  ${WHITE}Completed:${RESET}  ${BOLD}${COMPLETED}${RESET} / ${TOTAL}              "
    echo -e "  ${WHITE}Elapsed:${RESET}    ${ELAPSED_STR}              "
    echo -e "  ${WHITE}ETA:${RESET}        ${ETA_STR}              "
    echo -e "  ${WHITE}Last done:${RESET}  ${DIM}${LAST_TRACK}${RESET}              "

    if [ "$COMPLETED" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo ""
        echo -e "  ${GREEN}${BOLD}✅ ALL TRACKS COMPLETE${RESET}              "
        echo ""
        break
    fi

    sleep 3
done
