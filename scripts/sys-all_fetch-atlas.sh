#!/bin/bash

# Use the first argument as PROJECT_ROOT, or assume parent dir if run manually from /scripts
PROJECT_ROOT="${1:-$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}"
CONFIG_PATH="$PROJECT_ROOT/config.toml"

# Use exported ATLAS_DIR or default to a folder in project root
ATLAS_DIR="${ATLAS_DIR:-$PROJECT_ROOT/atlases}"

echo "Starting download to $ATLAS_DIR..."
mkdir -p "$ATLAS_DIR"

KEYS=("cortical_atlas" "cortical_atlas_lut" "subcortical_atlas" "subcortical_atlas_lut")

for key in "${KEYS[@]}"; do
    case "$key" in
        "cortical_atlas") URL="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/MNI/yeo17/400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz" ;;
        "cortical_atlas_lut") URL="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/HCP/fsLR32k/yeo17/400Parcels_Yeo2011_17Networks_info.txt" ;;
        "subcortical_atlas") URL="https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S3_3T.nii" ;;
        "subcortical_atlas_lut") URL="https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_label.txt" ;;
    esac

    FILENAME=$(basename "$URL")
    DEST="$ATLAS_DIR/$FILENAME"
    
    if [ ! -f "$DEST" ]; then
        echo "Downloading $FILENAME..."
        curl -L "$URL" -o "$DEST"
    else
        echo "$FILENAME already exists. Skip download."
    fi

    if [ -f "$CONFIG_PATH" ]; then
        ABS_PATH=$(realpath "$DEST")
        echo "Updating $key in config.toml..."
        sed -i.bak "s|^$key = .*|$key = \"$ABS_PATH\"|g" "$CONFIG_PATH"
        rm -f "${CONFIG_PATH}.bak"
    else
        echo "Warning: $CONFIG_PATH not found."
    fi
done

echo "Atlas fetching complete."