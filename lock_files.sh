#!/bin/bash
#
# lock_files.sh - Lock critical AOR-MIR files with immutability flags
# Makes specified files immutable and creates integrity checksums
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the sudo helper
source "$SCRIPT_DIR/sudo_helper.sh"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Files to protect
FILES_TO_LOCK=(
    "$SCRIPT_DIR/aor_mir.py"
    "$SCRIPT_DIR/aave_lexicon.json"
)

# Checksum file location
CHECKSUM_FILE="$SCRIPT_DIR/.file_checksums.sha256"

# Function to display status header
show_header() {
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}  AOR-MIR File Protection Utility${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
}

# Function to check if a file exists
check_file_exists() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo -e "${RED}ERROR: File not found: $file${NC}"
        return 1
    fi
    return 0
}

# Function to check if a file is already locked
is_file_locked() {
    local file="$1"
    local flags=$(ls -lO "$file" 2>/dev/null | awk '{print $5}')

    if [[ "$flags" == *"uchg"* ]]; then
        return 0  # File is locked
    fi
    return 1  # File is not locked
}

# Function to unlock a file (needed to modify it)
unlock_file() {
    local file="$1"
    sudo chflags nouchg "$file" 2>/dev/null
    return $?
}

# Function to lock a file with immutable flag
lock_file() {
    local file="$1"
    sudo chflags uchg "$file" 2>/dev/null
    return $?
}

# Function to generate SHA256 checksum for a file
generate_checksum() {
    local file="$1"
    shasum -a 256 "$file" 2>/dev/null
}

# Function to verify file integrity
verify_checksum() {
    local file="$1"
    local stored_checksum="$2"
    local current_checksum=$(shasum -a 256 "$file" 2>/dev/null | awk '{print $1}')

    if [ "$current_checksum" == "$stored_checksum" ]; then
        return 0
    fi
    return 1
}

# Function to show macOS confirmation dialog
show_confirmation() {
    local files_list=""
    for file in "${FILES_TO_LOCK[@]}"; do
        files_list="$files_list\\n  - $(basename "$file")"
    done
    files_list="$files_list\\n  - $(basename "$CHECKSUM_FILE")"

    osascript -e "
        display dialog \"The following files will be made immutable (locked):$files_list\\n\\nOnce locked, these files cannot be modified or deleted without first unlocking them with administrator privileges.\\n\\nDo you want to continue?\" with title \"Confirm File Protection\" buttons {\"Cancel\", \"Lock Files\"} default button \"Lock Files\" with icon caution
    " 2>/dev/null

    return $?
}

# Function to show completion dialog
show_completion_dialog() {
    local success_count="$1"
    local total_count="$2"

    osascript -e "
        display dialog \"File protection complete!\\n\\n$success_count of $total_count files have been locked and protected.\\n\\nA SHA256 checksum file has been created for integrity verification.\" with title \"Protection Complete\" buttons {\"OK\"} default button \"OK\" with icon note
    " 2>/dev/null
}

# Main function
main() {
    show_header

    echo -e "${BLUE}Checking files to protect...${NC}"
    echo ""

    # Verify all files exist
    local missing_files=0
    for file in "${FILES_TO_LOCK[@]}"; do
        if check_file_exists "$file"; then
            echo -e "  ${GREEN}[FOUND]${NC} $(basename "$file")"
        else
            missing_files=$((missing_files + 1))
        fi
    done

    if [ $missing_files -gt 0 ]; then
        echo ""
        echo -e "${RED}Cannot proceed: $missing_files file(s) not found.${NC}"
        show_error_dialog "One or more required files are missing. Please ensure all files exist before running this script."
        exit 1
    fi

    echo ""

    # Show confirmation dialog
    echo -e "${YELLOW}Requesting user confirmation...${NC}"
    if ! show_confirmation; then
        echo -e "${RED}Operation cancelled by user.${NC}"
        exit 1
    fi

    echo -e "${GREEN}User confirmed. Proceeding...${NC}"
    echo ""

    # Authenticate with sudo
    echo -e "${BLUE}Step 1: Obtaining administrator privileges...${NC}"
    if ! authenticate_sudo; then
        echo -e "${RED}Failed to obtain administrator privileges. Exiting.${NC}"
        exit 1
    fi
    echo ""

    # Generate checksums
    echo -e "${BLUE}Step 2: Generating integrity checksums...${NC}"

    # If checksum file exists and is locked, unlock it first
    if [ -f "$CHECKSUM_FILE" ] && is_file_locked "$CHECKSUM_FILE"; then
        echo -e "  ${YELLOW}Unlocking existing checksum file...${NC}"
        unlock_file "$CHECKSUM_FILE"
    fi

    # Create/overwrite checksum file
    echo "# AOR-MIR File Integrity Checksums" > "$CHECKSUM_FILE"
    echo "# Generated: $(date)" >> "$CHECKSUM_FILE"
    echo "# DO NOT MODIFY THIS FILE" >> "$CHECKSUM_FILE"
    echo "" >> "$CHECKSUM_FILE"

    for file in "${FILES_TO_LOCK[@]}"; do
        # Unlock file if already locked (to ensure we can read it properly)
        if is_file_locked "$file"; then
            echo -e "  ${YELLOW}Temporarily unlocking: $(basename "$file")${NC}"
            unlock_file "$file"
        fi

        # Generate and store checksum
        local checksum=$(generate_checksum "$file")
        echo "$checksum" >> "$CHECKSUM_FILE"
        echo -e "  ${GREEN}[CHECKSUM]${NC} $(basename "$file")"
    done
    echo ""

    # Lock the files
    echo -e "${BLUE}Step 3: Applying immutable flags...${NC}"
    local locked_count=0
    local total_count=${#FILES_TO_LOCK[@]}

    for file in "${FILES_TO_LOCK[@]}"; do
        if lock_file "$file"; then
            echo -e "  ${GREEN}[LOCKED]${NC} $(basename "$file")"
            locked_count=$((locked_count + 1))
        else
            echo -e "  ${RED}[FAILED]${NC} $(basename "$file")"
        fi
    done

    # Lock the checksum file
    echo ""
    echo -e "${BLUE}Step 4: Protecting checksum file...${NC}"
    if lock_file "$CHECKSUM_FILE"; then
        echo -e "  ${GREEN}[LOCKED]${NC} $(basename "$CHECKSUM_FILE")"
        locked_count=$((locked_count + 1))
        total_count=$((total_count + 1))
    else
        echo -e "  ${RED}[FAILED]${NC} $(basename "$CHECKSUM_FILE")"
        total_count=$((total_count + 1))
    fi

    # Clear sudo credentials
    echo ""
    echo -e "${BLUE}Step 5: Cleaning up...${NC}"
    invalidate_sudo

    # Show summary
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}  Protection Summary${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
    echo -e "  Files processed: $total_count"
    echo -e "  Files locked:    $locked_count"
    echo -e "  Checksum file:   $CHECKSUM_FILE"
    echo ""

    if [ $locked_count -eq $total_count ]; then
        echo -e "${GREEN}All files have been successfully protected!${NC}"
        show_completion_dialog "$locked_count" "$total_count"
    else
        echo -e "${YELLOW}Warning: Some files could not be locked.${NC}"
    fi

    echo ""
    echo -e "${BLUE}To unlock files later, use:${NC}"
    echo -e "  sudo chflags nouchg <filename>"
    echo ""
}

# Run main function
main "$@"
