# TCP fMRI Preprocessing Pipeline

Rust fMRI preprocessing pipeline for TCP (Transdiagnostic Connectomes Project) dataset. 

## Pipeline Stages

Execution order (crates 00-09):
1. **00tcp_subject_selection**: Filter subjects
2. **01fmri_parcellation**: Parcellate regions
3. **02fmri_segment_trials**: Segment timeseries
4. **03cwt**: Wavelet transform
5. **04mvmd**: Multivariate Mode Decomposition 
6. **05hilbert**: Hilbert transform
7. **06fc**: Functional connectivity
8. **07feature_extraction**: Extract features (CNN)
9. **08data_splitting**: Train/test split
10. **09classification**: Model classification

**utils**: Shared helpers.
**cli**: Pipeline CLI.

## Prerequisites
Rust 1.82.0+. Git-annex for large data. HDF5 for timeseries.

## Setup & Initialization

Run main initialization script to prepare paths, atlases, and system environment.

```bash
# Initialize project paths, download atlases, and build dependencies
bash scripts/init.sh
```

### IDUN Cluster Setup

IDUN needs specific modules and environment prep. The `init.sh` script handles this if it detects IDUN. Trigger cluster mode by passing `idun` argument OR creating `.sys-idun` file.

```bash
# Option 1: Pass idun argument
bash scripts/init.sh idun

# Option 2: Empty trigger file
touch .sys-idun
bash scripts/init.sh
```
This auto-loads IDUN specific config defaults and builds HDF5 (if missing). 

**CRITICAL (IDUN):** After running `init.sh`, you must source the environment script to load Rust and CUDA modules into your current shell session:
```bash
source scripts/sys-idun_env.sh
```

### Local Machine Setup

**CRITICAL:** Run `bash scripts/init.sh` on local (Mac/Windows/Linux). It copies `config.toml` and fetches atlases. But **you MUST manually edit `config.toml`** and set your local directory paths (`tcp_repo_dir`, `fmriprep_output_dir`, etc.) after script finishes.

## Building

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -p cli -- --help
cargo run --release -p cli -- tcp-select-subjects
```

## Configuration

For local machines, copy example config if `init.sh` didn't. Set paths for platform manually.

```bash
cp config.toml.example config.toml
```

Example local paths:
- `tcp_repo_dir` = `/Users/ipeglin/data/ds005237`
- `fmriprep_output_dir` = `/Users/ipeglin/data/fmriprep`
