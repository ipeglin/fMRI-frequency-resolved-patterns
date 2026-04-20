#!/bin/bash
# init.sh - Main initialization script for the TCP fMRI pipeline.
#
# Sub-scripts executed:
# - sys-idun_config.sh: Auto-fills config.toml with IDUN-specific paths.
# - sys-all_fetch-atlas.sh: Downloads required brain atlases and updates config.toml.
# - sys-idun_build-hdf5.sh: Compiles HDF5 from source (if missing on IDUN).
# - sys-idun_env.sh: Loads modules (Rust, CUDA) and sets ENV vars (HDF5_DIR, LIBTORCH).
# - sys-local_install-deps.sh: Attempts to install HDF5 via brew/apt/dnf on local setup.
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

# --- 2. System Detection ---
IS_IDUN=false
if [[ "$1" == "idun" ]] || [[ -f "$PROJECT_ROOT/.sys-idun" ]]; then
    IS_IDUN=true
    echo ">> System detected: IDUN Cluster"
else
    echo ">> System detected: Local Machine"
fi

# --- 3. Initialize Config ---
CONFIG_FILE="$PROJECT_ROOT/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo ">> Initializing config.toml..."
    cp "$PROJECT_ROOT/config.toml.example" "$CONFIG_FILE"

    if [ "$IS_IDUN" = true ]; then
        IDUN_CFG_SCRIPT="$SCRIPTS_DIR/sys-idun_config.sh"
        if [ -f "$IDUN_CFG_SCRIPT" ]; then
            chmod +x "$IDUN_CFG_SCRIPT"
            bash "$IDUN_CFG_SCRIPT" "$CONFIG_FILE"
        fi
    else
        echo ">> Configuring local default paths..."
        OS="$(uname -s)"
        if [[ "$OS" == "Linux"* ]]; then
            DL_DIR="$HOME/downloads/ds005237"
        else
            DL_DIR="$HOME/Downloads/ds005237"
        fi
        sed -i.bak -e "s|^tcp_repo_dir = .*|tcp_repo_dir = \"$DL_DIR\"|" "$CONFIG_FILE"
        rm -f "${CONFIG_FILE}.bak"
    fi
else
    echo ">> config.toml already exists."
fi

# --- 4. Define ATLAS_DIR ---
if [ "$IS_IDUN" = true ]; then
    export ATLAS_DIR="/cluster/work/$USER/atlases"
else
    DEFAULT_LOCAL_DIR="$PROJECT_ROOT/atlases"
    printf "Enter atlas directory [Default: %s]: " "$DEFAULT_LOCAL_DIR"
    read USER_INPUT
    export ATLAS_DIR="${USER_INPUT:-$DEFAULT_LOCAL_DIR}"
fi
echo ">> ATLAS_DIR set to: $ATLAS_DIR"

# --- 5. Run Atlas Fetcher ---
FETCH_SCRIPT="$SCRIPTS_DIR/sys-all_fetch-atlas.sh"
if [ -f "$FETCH_SCRIPT" ]; then
    chmod +x "$FETCH_SCRIPT"
    # We pass the project root as an argument so the fetcher knows where to find config.toml
    bash "$FETCH_SCRIPT" "$PROJECT_ROOT"
else
    echo "!! Error: $FETCH_SCRIPT not found."
fi

# --- 6. HDF5 & Environment Setup ---
if [ "$IS_IDUN" = true ]; then
    HDF5_INSTALL_DIR="$HOME/hdf5"
    if [ ! -f "$HDF5_INSTALL_DIR/lib/libhdf5.so" ]; then
        echo ">> HDF5 not found. Building explicitly for IDUN..."
        BUILD_SCRIPT="$SCRIPTS_DIR/sys-idun_build-hdf5.sh"
        if [ -f "$BUILD_SCRIPT" ]; then
            bash "$BUILD_SCRIPT"
        fi
    fi

    ENV_SCRIPT="$SCRIPTS_DIR/sys-idun_env.sh"
    if [ -f "$ENV_SCRIPT" ]; then
        source "$ENV_SCRIPT"
    fi
    
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        echo ""
        echo "!! WARNING: SCRIPT NOT SOURCED !!"
        echo "Because you ran this with 'bash' instead of 'source', the Rust/CUDA"
        echo "modules and IDUN environment variables will NOT persist in your terminal."
        echo "To apply them to your current session, run:"
        echo "    source $ENV_SCRIPT"
        echo ""
    fi
else
    # Local OS dependency install
    DEPS_SCRIPT="$SCRIPTS_DIR/sys-local_install-deps.sh"
    if [ -f "$DEPS_SCRIPT" ]; then
        chmod +x "$DEPS_SCRIPT"
        bash "$DEPS_SCRIPT"
    else
        if ! command -v h5cc >/dev/null 2>&1; then
            echo "!! Warning: HDF5 (h5cc) not found. You may need to install it manually."
        fi
    fi
fi

echo "--- Initialization Complete ---"