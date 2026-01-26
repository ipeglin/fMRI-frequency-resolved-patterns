# TCP fMRI Preprocessing Pipeline

A Rust-based fMRI preprocessing pipeline for the TCP (Transdiagnostic Connectomes Project) dataset.

## Prerequisites

### Rust Toolchain

Rust 1.82.0 or later is required (for Rust 2024 edition):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Git-annex

Required for fetching large data files from the TCP dataset:

```bash
# macOS
brew install git-annex

# Ubuntu/Debian
sudo apt-get install git-annex
```

### HDF5

Required for writing timeseries output files:

```bash
# macOS
brew install hdf5

# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# Fedora/RHEL
sudo dnf install hdf5-devel
```

## Building

```bash
cargo build --release
```

## Usage

```bash
# Show help
cargo run --release -p cli -- --help

# Run subject selection
cargo run --release -p cli -- tcp-select-subjects

# Run fMRI preprocessing
cargo run --release -p cli -- tcp-fmri-preprocess
```

## Configuration

Copy the example configuration and edit as needed:

```bash
cp config.toml.example config.toml
```
