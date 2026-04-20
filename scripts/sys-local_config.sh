#!/bin/bash
# sys-local_config.sh - Set local default paths in config.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

CONFIG_FILE="$1"
if [ ! -f "$CONFIG_FILE" ]; then
    log_err "config file $CONFIG_FILE not found."
    exit 1
fi

log_info "Configuring local specific paths in config.toml"

OS="$(uname -s)"
if [[ "$OS" == "Linux"* ]]; then
    DL_DIR="$HOME/downloads/ds005237"
else
    DL_DIR="$HOME/Downloads/ds005237"
fi

# Only replace if empty to avoid overwriting user paths on local machines
sed -i.bak -e "s|^tcp_repo_dir = \"\"|tcp_repo_dir = \"$DL_DIR\"|" "$CONFIG_FILE"

rm -f "${CONFIG_FILE}.bak"
log_success "Local config defaults applied."
