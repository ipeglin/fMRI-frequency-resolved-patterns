#!/bin/bash
# sys-all_fetch-weights.sh - Fetch CNN weights and update config.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

PROJECT_ROOT="$1"
WEIGHTS_DIR="$2"
CONFIG_FILE="$PROJECT_ROOT/config.toml"

# --- CONFIGURATION ---
GH_USER="ipeglin"
GH_REPO="masters-thesis-supplementary"
GH_BRANCH="main"
WEIGHTS_FILE_PATH="data/processed/densenet201_imagenet.safetensors"

if [ ! -f "$CONFIG_FILE" ]; then
    log_err "config file $CONFIG_FILE not found."
    exit 1
fi

mkdir -p "$WEIGHTS_DIR"

# DenseNet201 Weights URL constructed using variables
DENSENET_URL="https://github.com/$GH_USER/$GH_REPO/raw/$GH_BRANCH/$WEIGHTS_FILE_PATH"
DENSENET_FILE="$WEIGHTS_DIR/$(basename "$WEIGHTS_FILE_PATH")"

log_info "CNN Weights output directory set to: $WEIGHTS_DIR"

if [ ! -f "$DENSENET_FILE" ]; then
    log_info "Downloading densenet201_imagenet.safetensors weights..."
    curl -L -s -f "$DENSENET_URL" -o "$DENSENET_FILE"

    if [ $? -eq 0 ]; then
        log_success "Weights downloaded successfully."
    else
        log_err "Failed to download weights."
        # Remove the potentially corrupt output file if curl failed midway
        rm -f "$DENSENET_FILE"
        exit 1
    fi
else
    log_info "densenet201_imagenet.safetensors already exists, skipping download."
fi

# Update config.toml
sed -i.bak -e "s|^cnn_weights_path = \"\"|cnn_weights_path = \"$DENSENET_FILE\"|" "$CONFIG_FILE"
rm -f "${CONFIG_FILE}.bak"

log_success "CNN_weights_path configured in config.toml."
