#!/bin/bash
# sys-all_fetch-atlas.sh - Fetch atlases and update config.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

# --- CONFIGURATION ---
# Repo 1: ThomasYeoLab/CBIG
YEO_USER="ThomasYeoLab"
YEO_REPO="CBIG"
YEO_BRANCH="master"
YEO_BASE="https://raw.githubusercontent.com/$YEO_USER/$YEO_REPO/$YEO_BRANCH"

# Repo 2: yetianmed/subcortex
TIAN_USER="yetianmed"
TIAN_REPO="subcortex"
TIAN_BRANCH="master"
TIAN_BASE="https://raw.githubusercontent.com/$TIAN_USER/$TIAN_REPO/$TIAN_BRANCH"

# Use the first argument as PROJECT_ROOT, or assume parent dir if run manually from /scripts
PROJECT_ROOT="${1:-$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}"
CONFIG_PATH="$PROJECT_ROOT/config.toml"

# Use exported ATLAS_DIR or default to a folder in project root
ATLAS_DIR="${ATLAS_DIR:-$PROJECT_ROOT/atlases}"

log_info "Target directory: $ATLAS_DIR"
mkdir -p "$ATLAS_DIR"

KEYS=("cortical_atlas" "cortical_atlas_lut" "subcortical_atlas" "subcortical_atlas_lut")

for key in "${KEYS[@]}"; do
    case "$key" in
        "cortical_atlas")
            URL="$YEO_BASE/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/MNI/yeo17/400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz"
            ;;
        "cortical_atlas_lut")
            URL="$YEO_BASE/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/HCP/fsLR32k/yeo17/400Parcels_Yeo2011_17Networks_info.txt"
            ;;
        "subcortical_atlas")
            URL="$TIAN_BASE/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T.nii"
            ;;
        "subcortical_atlas_lut")
            URL="$TIAN_BASE/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_label.txt"
            ;;
    esac

    FILENAME=$(basename "$URL")
    DEST="$ATLAS_DIR/$FILENAME"

    if [ ! -f "$DEST" ]; then
        log_info "Downloading $FILENAME..."
        curl -s -L "$URL" -o "$DEST"
        log_success "Downloaded $FILENAME"
    else
        log_info "$FILENAME already exists. Skip download."
    fi

    if [ -f "$CONFIG_PATH" ]; then
        ABS_PATH=$(realpath "$DEST")
        log_info "Updating $key in config.toml"
        sed -i.bak "s|^$key = .*|$key = \"$ABS_PATH\"|g" "$CONFIG_PATH"
        rm -f "${CONFIG_PATH}.bak"
    else
        log_warn "$CONFIG_PATH not found."
    fi
done

log_success "Atlas fetching complete."
