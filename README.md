# 🧠 TCP fMRI Preprocessing Pipeline

A high-performance Rust-based preprocessing pipeline for the **Transdiagnostic Connectomes Project (TCP)** dataset.

---

## 🏗️ Pipeline Stages

The pipeline consists of 9 sequential stages (crates `00`–`08`). **Execution order is mandatory** as each stage depends on the output of the previous one:

1.  **`00tcp_subject_selection`**: Filter subjects based on criteria.
2.  **`01fmri_parcellation`**: Parcellate brain regions.
3.  **`02fmri_segment_trials`**: Segment timeseries data.
4.  **`03cwt`**: Perform Wavelet transform.
5.  **`04mvmd`**: Multivariate Mode Decomposition.
6.  **`05hilbert`**: Apply Hilbert transform.
7.  **`06fc`**: Compute Functional Connectivity.
8.  **`07feature_extraction`**: Extract features using CNNs.
9.  **`08classification`**: Perform model classification.

* **`utils`**: Shared helper functions.
* **`cli`**: Main Pipeline Command Line Interface.

---

## 📋 Prerequisites

* **Rust**: Version 1.82.0+.
* **Git-annex**: For managing large datasets.
* **HDF5**: Required for timeseries storage.
* **Libtorch**: PyTorch C++ API for deep learning features (Stage 07).

---

## 🛠️ Setup & Initialization

Run the main initialization script to prepare paths, download required atlases, and configure the environment.

```bash
bash ./scripts/init.sh
```

### 🏫 IDUN Cluster Setup
The script also provides IDUN-specific setup such as handling modules and building HDF5. To enable this, either pass `idun` as an argument or create a trigger file named `.sys-idun` in the project root before running the init script:

```bash
# Option 1: Pass idun argument
bash ./scripts/init.sh idun

# Option 2: Use a trigger file
touch .sys-idun && bash ./scripts/init.sh
```

> [!IMPORTANT]
> **Post-Init Action (IDUN):** After initialization, you **must** source the environment script to load Rust and CUDA modules into your current session:
> ```bash
> source ./scripts/sys-idun_env.sh
> ```
> This script also downloads `libtorch` to `$HOME/libtorch` if it is not found in your environment.

### 💻 Local Machine Setup

> [!CAUTION]
> **Manual Config Required:** On local machines, you **must** edit `config.toml` after running the init script. Set your local directory paths for `tcp_repo_dir`, `fmriprep_output_dir`, etc. manually.

**Manual Libtorch Installation:**
1. Download [LibTorch](https://pytorch.org/) (match your system: CPU, MPS, or CUDA).
2. Extract to a known path (e.g., `$HOME/libtorch`).
3. Set your environment variables:
   ```bash
   export LIBTORCH=$HOME/libtorch
   # macOS:
   export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH 
   # Linux:
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
   ```

---

## 🔨 Building

Compile the entire workspace in release mode for optimal performance:

```bash
cargo build --release
```

---

## 🚀 Usage

### Running on IDUN using Slurm (Recommended)
For significantly improved processing runtime, use the preconfigured Slurm schemas. These are generated during initialization and include automatic NTNU username injection.

These files are installed exclusively on IDUN, and will not be present on local machines after initialization. If you are on IDUN, you should run the full pipeline with e.g.:

```bash
sbatch ./slurm/run_pipeline.slurm
```

> [!WARNING]
> Rerunning `scripts/init.sh` may overwrite your Slurm configurations. If you modify files in `./slurm/`, rename them or move them to a new directory to prevent loss.

### Running the pipeline locally
To execute the full automated pipeline:
```bash
bash scripts/run-pipeline.sh
```

### Single-crate execution
> [!IMPORTANT]
> You are responsible for executing steps in the correct order when running crates individually.

```bash
# Example: Run subject selection
cargo run -- select-subjects

# Example: Force recomputation
cargo run -- parcellate-bold --force 
```

| Crate                    | CLI Command           |
| :---                     | :---                  |
| `00subject_selection`    | `select-subjects`     |
| `01fmri_parcellation`    | `parcellate-bold`     |
| `02fmri_segment_trials`  | `segment-trials`      |
| `03cwt`                  | `cwt`                 |
| `04mvmd`                 | `mvmd`                |
| `05hilbert`              | `hht`                 |
| `06fc`                   | `fc`                  |
| `07feature_extraction`   | `feature-extraction`  |
| `08classification`       | `classify`            |

---

## ⚙️ Configuration

If `init.sh` does not generate a configuration file, copy the example manually:

```bash
cp config.toml.example config.toml
```

**Required Local Paths:**
* `tcp_repo_dir`: Path to your dataset (e.g., `/Users/name/data/ds005237`).
* `fmriprep_output_dir`: Path to fMRIPrep outputs.
