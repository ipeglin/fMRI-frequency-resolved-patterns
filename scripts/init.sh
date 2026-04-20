#!/bin/bash
# init.sh - Main initialization script for the TCP fMRI pipeline.
#
# Sub-scripts executed (or prompted):
# - sys-idun_config.sh: Auto-fills config.toml with IDUN-specific paths.
# - sys-local_config.sh: Auto-fills config.toml with Local-specific paths (e.g. Downloads dir).
# - sys-all_fetch-atlas.sh: Downloads required brain atlases and updates config.toml.
# - sys-idun_build-hdf5.sh: Compiles HDF5 from source (if missing on IDUN).
# - sys-local_install-deps.sh: Attempts to install HDF5 via brew/apt/dnf on local setup.
#   (Instructs to source sys-idun_env.sh afterwards)
#
# NOTE FOR LOCAL MACHINES: You MUST manually edit config.toml to set your specific paths after running this script.

# --- 1. Path Setup ---
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we are running from /scripts or the root
if [[ "$SCRIPT_DIR" == */scripts ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    SCRIPTS_DIR="$SCRIPT_DIR"
else
    PROJECT_ROOT="$SCRIPT_DIR"
    SCRIPTS_DIR="$PROJECT_ROOT/scripts"
fi

source "$SCRIPTS_DIR/sys-logger.sh"
log_title "TCP fMRI Preprocessing Setup"

# --- 2. System Detection ---
IS_IDUN=false
if [[ "$1" == "idun" ]] || [[ -f "$PROJECT_ROOT/.sys-idun" ]]; then
    IS_IDUN=true
    log_info "System detected: IDUN Cluster"
else
    log_info "System detected: Local Machine"
fi

# --- 3. Initialize Config ---
log_step "Initializing config.toml"
CONFIG_FILE="$PROJECT_ROOT/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    log_info "Copying config.toml.example to config.toml"
    cp "$PROJECT_ROOT/config.toml.example" "$CONFIG_FILE"
else
    log_info "config.toml already exists"
fi

if [ "$IS_IDUN" = true ]; then
    IDUN_CFG_SCRIPT="$SCRIPTS_DIR/sys-idun_config.sh"
    if [ -f "$IDUN_CFG_SCRIPT" ]; then
        chmod +x "$IDUN_CFG_SCRIPT"
        bash "$IDUN_CFG_SCRIPT" "$CONFIG_FILE"
    fi
else
        LOCAL_CFG_SCRIPT="$SCRIPTS_DIR/sys-local_config.sh"
        if [ -f "$LOCAL_CFG_SCRIPT" ]; then
            chmod +x "$LOCAL_CFG_SCRIPT"
            bash "$LOCAL_CFG_SCRIPT" "$CONFIG_FILE"
        fi
    DEFAULT_LOCAL_DIR="$PROJECT_ROOT/atlases"
    printf "  ${C_CYAN}ℹ${C_RESET} Enter atlas directory [Default: %s]: " "$DEFAULT_LOCAL_DIR"
    read USER_INPUT
    export ATLAS_DIR="${USER_INPUT:-$DEFAULT_LOCAL_DIR}"
fi
log_success "ATLAS_DIR set to: $ATLAS_DIR"

# --- 5. Run Atlas Fetcher ---
log_step "Fetching Brain Atlases"
FETCH_SCRIPT="$SCRIPTS_DIR/sys-all_fetch-atlas.sh"
if [ -f "$FETCH_SCRIPT" ]; then
    chmod +x "$FETCH_SCRIPT"
    # We pass the project root as an argument so the fetcher knows where to find config.toml
    bash "$FETCH_SCRIPT" "$PROJECT_ROOT"
else
    log_err "$FETCH_SCRIPT not found."
fi

# --- 6. HDF5 & Environment Setup ---
log_step "Setting up Environment"
if [ "$IS_IDUN" = true ]; then
    HDF5_INSTALL_DIR="$HOME/hdf5"
    if [ ! -f "$HDF5_INSTALL_DIR/lib/libhdf5.so" ]; then
        log_warn "HDF5 not found. Building explicitly for IDUN..."
        BUILD_SCRIPT="$SCRIPTS_DIR/sys-idun_build-hdf5.sh"
        if [ -f "$BUILD_SCRIPT" ]; then
            bash "$BUILD_SCRIPT"
        fi
    fi

    ENV_SCRIPT="$SCRIPTS_DIR/sys-idun_env.sh"
    echo ""
    log_warn "IDUN ENVIRONMENT REQUIRES ACTIVATION"
    log_info "To load Rust/CUDA modules and set HDF5 paths for this session, you MUST run:"
    echo -e "    ${C_BOLD}${C_CYAN}source $ENV_SCRIPT${C_RESET}"
    echo ""
else
    # Local OS dependency install
    DEPS_SCRIPT="$SCRIPTS_DIR/sys-local_install-deps.sh"
    if [ -f "$DEPS_SCRIPT" ]; then
        chmod +x "$DEPS_SCRIPT"
        bash "$DEPS_SCRIPT"
    else
        if ! command -v h5cc >/dev/null 2>&1; then
            log_warn "HDF5 (h5cc) not found. You may need to install it manually."
        fi
    fi
fi

log_title "Initialization Complete"