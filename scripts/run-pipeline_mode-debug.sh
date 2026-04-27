#!/bin/bash
# sys-all_run-pipeline.sh - System-wide script to build the project and run the full pipeline
#
# Runs the CLI commands in the specified order:
# 00. select-subjects
# 01. parcellate-bold
# 02. segment-trials
# 03. cwt
# 04. decompose-mvmd
# 05. hht
# 06. fc
# 07. feature-extraction
# 08. split-data
# 09. classify

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/sys-logger.sh"
log_title "Running Full TCP fMRI Pipeline"

# --- Environment Check for IDUN ---
IS_IDUN=false
if [[ -f "$PROJECT_ROOT/.sys-idun" ]]; then
    IS_IDUN=true
fi

if [ "$IS_IDUN" = true ]; then
    log_info "IDUN System detected. Checking environment..."
    # A simple way to check if sys-idun_env.sh was sourced is to check for variables it sets
    # like HDF5_DIR or LIBTORCH, or to check if the 'rustc' command is available.
    if [[ -z "$HDF5_DIR" ]] || [[ -z "$LIBTORCH" ]] || ! command -v rustc >/dev/null 2>&1; then
        log_err "CRITICAL: It appears the IDUN environment has not been sourced."
        log_err "Please run 'source scripts/sys-idun_env.sh' before continuing."
        exit 1
    fi
    log_success "IDUN environment checks passed."
fi

# Apply LIBTORCH paths to dynamic linker for macOS / Linux
if [ -n "$LIBTORCH" ]; then
    export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
fi

# --- Build Project ---
log_step "Building the project (cargo build --release --all-features)"
cd "$PROJECT_ROOT"
if ! cargo build --release --all-features; then
    log_err "Failed to build the project. Aborting pipeline."
    exit 1
fi
log_success "Build complete."

# --- Pipeline Execution ---
# Using cargo run handles the dynamic library paths (DYLD_LIBRARY_PATH/LD_LIBRARY_PATH)
# automatically on macOS/Linux for bindings like libtorch.

run_stage() {
    local stage_num=$1
    local subcommand=$2
    shift 2
    local args=("$@")

    log_step "Running Stage $stage_num: $subcommand"
    if cargo run --release --all-features -- "$subcommand" "${args[@]}" --log-level debug; then
        log_success "Stage $stage_num ($subcommand) completed successfully."
    else
        log_err "Stage $stage_num ($subcommand) failed! Aborting pipeline."
        exit 1
    fi
}

# The pipeline execution logic
# You can append extra global arguments (like --config) here if needed.
# e.g., run_stage "00" "select-subjects" "--config" "config.toml"

run_stage "00" "select-subjects"
run_stage "01" "parcellate-bold"
run_stage "02" "segment-trials"
run_stage "03" "cwt"
run_stage "04" "mvmd"
run_stage "05" "hht"
run_stage "06" "fc"
run_stage "07" "feature-extraction"
run_stage "08" "classify"

log_title "Pipeline Execution Finished Successfully"
