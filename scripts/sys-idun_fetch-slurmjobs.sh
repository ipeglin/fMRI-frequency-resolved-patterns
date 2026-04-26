#!/bin/bash
# sys-idun_fetch-slurmjobs.sh - Fetch and configure slurm job schemas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

PROJECT_ROOT="$1"
SLURM_DEST_DIR="$2"
# Get the current local username
CURRENT_USER=$(whoami)

# --- JQ INSTALLATION ---
if ! command -v jq &> /dev/null; then
    log_info "jq not found. Installing..."
    sudo yum install epel-release -y && sudo yum install jq -y
fi

# --- CONFIGURATION ---
GH_USER="ipeglin"
GH_REPO="masters-thesis-supplementary"
GH_REMOTE_DIR="slurm"

# SCALABLE INJECTION CONFIG
declare -A REPLACEMENTS
REPLACEMENTS["PROJECT_ROOT"]="$PROJECT_ROOT"
REPLACEMENTS["USER_NAME"]="$CURRENT_USER"

if [ -z "$PROJECT_ROOT" ] || [ -z "$SLURM_DEST_DIR" ]; then
    log_err "Usage: $0 <PROJECT_ROOT> <SLURM_DEST_DIR>"
    exit 1
fi

# 1. Fetch File List
API_URL="https://api.github.com/repos/$GH_USER/$GH_REPO/contents/$GH_REMOTE_DIR"
log_info "Fetching file list from GitHub..."
FILES_JSON=$(curl -s -f -H "Cache-Control: no-cache" "$API_URL")

if [ $? -ne 0 ]; then
    log_err "Failed to fetch directory listing."
    exit 1
fi

while read -r DOWNLOAD_URL; do
    if [ -n "$DOWNLOAD_URL" ] && [ "$DOWNLOAD_URL" != "null" ]; then
        if [ ! -d "$SLURM_DEST_DIR" ]; then mkdir -p "$SLURM_DEST_DIR"; fi

        FILENAME=$(basename "$DOWNLOAD_URL")
        TARGET_FILE="$SLURM_DEST_DIR/$FILENAME"

        # Optional: Remove old local version before downloading to ensure a clean state
        rm -f "$TARGET_FILE"

        log_info "Downloading and configuring $FILENAME..."
        curl -L -s -f -H "Cache-Control: no-cache" "$DOWNLOAD_URL" -o "$TARGET_FILE"

        if [ $? -eq 0 ]; then
            # --- SCALABLE INJECTION ---
            for KEY in "${!REPLACEMENTS[@]}"; do
                VALUE="${REPLACEMENTS[$KEY]}"
                # Replaces 'KEY = ""' or 'KEY=""' at start of line
                sed -i "s|^$KEY = .*|$KEY=\"$VALUE\"|g" "$TARGET_FILE"
                sed -i "s|^$KEY=.*|$KEY=\"$VALUE\"|g" "$TARGET_FILE"
            done

            # --- SPECIFIC FIX FOR SLURM HEADERS ---
            # Fixed: Added \b (word boundary) so $USER_NAME is NOT matched by $USER
            sed -i "s|\$USER\b|$CURRENT_USER|g" "$TARGET_FILE"

            # Additional robustness: handle ${USER} if used
            sed -i "s|\${USER}|$CURRENT_USER|g" "$TARGET_FILE"

            log_success "Configured $FILENAME for user $CURRENT_USER"
            chmod +x "$TARGET_FILE"
        fi
    fi
done < <(echo "$FILES_JSON" | jq -r '.[] | select(.type=="file") | .download_url')
