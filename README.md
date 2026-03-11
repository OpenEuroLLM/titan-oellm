# Titan-OELLM

A TorchTitan-based framework for training large language models on HPC systems.
Focus on research and deveoplent of new architectures and optimization methods. 

## Features

- **Custom Model Architectures**: Easily modiufy architecturers based on default implementations such as Qwen3
- **HPC Optimization**: SLURM scripts for different clusters
- **Flexible Configuration**: TOML-based configs with cluster-specific path resolution
- **Validation During Training**: Comprehensive validation with TensorBoard integration
- **Memory-Mapped Datasets**: Efficient data loading with chunking support



## Core Structure

- We use `torchtitan` as ist is.
- We build global custom components in `titan_oellm`
- We maintain private paths and configs in `user/***`


## Quick Start

### 1. Clone Repository

```bash
git clone --recursive <repo-url>
cd titan-oellm

# Load and verify TorchTitan submodule
git submodule update --init --recursive
cd torchtitan && git describe --tags && cd ..
# Should output: v0.2.1
```

### 2. Build Container

```bash
export APPTAINER_CACHEDIR=/path/to/your/cache
export APPTAINER_TMPDIR=/path/to/your/tmp
apptainer build --fakeroot titan_CLUSTER_0.2.1.sif titan_0.2.1.def

```

Test container:
```bash
apptainer shell --nv titan_CLUSTER_0.2.1.sif
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```


### 3. Set Up Your User Configuration

```bash
# Set your username (REQUIRED for all operations)
export TITAN_USER=your_username

# Create your config directory
mkdir -p user/$TITAN_USER

# Copy example templates
cp user/example/cluster_paths.toml.example user/$TITAN_USER/cluster_paths.toml

# Edit with your paths
# Replace <YOUR_PROJECT>, <YOUR_USER_ID> with your actual values
vim user/$TITAN_USER/cluster_paths.toml
```


### 4. Run Training

#### Local Testing (Recommended for Development)

```bash
export TITAN_USER=example

# Run locally (on your machine or interactive node)
bash submit_job.sh --local

# With custom dataset and config
DATASET=test_dataset TOKENIZER=neox CONFIG=user/example/configs/debug.toml bash submit_job.sh --local

# On cluster interactive node (after srun --pty bash)
CLUSTER=juwels DATASET=slimpajama_627b TOKENIZER=neox CONFIG=user/example/configs/debug.toml bash submit_job.sh --local
```

#### Cluster Submission (SLURM)

```bash
export TITAN_USER=example
export CLUSTER=juwels  # or capella, jupiter

# Submit training job (auto-selects slurm/<CLUSTER>.sh)
DATASET=fineweb_edu CONFIG=qwen3_custom.toml bash submit_job.sh

# Explicit script path also works
bash submit_job.sh slurm/juwels.sh
```

## Models

| Model | Description | Config |
|-------|-------------|--------|
| **gpt_plus** | GPT with QKNormPlus attention normalization and RoPE scaling | `base_plus.toml` |
| **qwen3_custom** | Qwen3 custom implementation with MoE support | `qwen3_custom.toml` |

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `TITAN_USER` | Username for user-specific configs | - | Yes |
| `CLUSTER` | Cluster name (local, juwels, capella, jupiter) | local (for --local), auto-detect (for SLURM) | No |
| `DATASET` | Dataset name from cluster_paths.toml | test_dataset (local), slimpajama_627b (SLURM) | No |
| `TOKENIZER` | Tokenizer name from cluster_paths.toml | neox | No |
| `CONFIG` | Config file path | user/$TITAN_USER/configs/debug.toml (local) | No |
| `NPROC` | Number of GPUs for local execution | 1 | No |

### Directory Structure

```
titan-oellm/
├── titan_oellm/           # Main source package
│   ├── models/            # Model implementations (gpt_plus, qwen3_custom)
│   ├── components/        # Training utilities (schedulers, validators)
│   ├── datasets/          # Dataloaders and tokenizers
│   └── configs/           # TOML configuration files
├── user/                  # User-specific configurations
│   └── example/           # Template configs (copy these)
├── scripts/               # Utility scripts
├── slurm/                 # SLURM job scripts per cluster
│   ├── juwels.sh
│   └── capella.sh
├── submit_job.sh          # Job submission wrapper
└── torchtitan/            # TorchTitan submodule (v0.2.0)
```

### User Configuration

Each user needs their own `user/<username>/cluster_paths.toml` with:

- **Cluster paths**: Output directories, cache locations
- **Tokenizer paths**: Per-tokenizer, per-cluster configurations  
- **Dataset paths**: Per-dataset, per-tokenizer, per-cluster configurations
- **Benchmark paths**: Optional evaluation benchmark locations

Copy template from `user/example/` and customize for your environment:

```bash
cp user/example/cluster_paths.toml.example user/myname/cluster_paths.toml
cp user/example/config.toml.example user/myname/configs/debug.toml
# Edit both files with your paths
```

Then run training:

```bash
TITAN_USER=myname CONFIG=user/myname/configs/debug.toml bash submit_job.sh --local
```
- **Tokenizers**: Paths to tokenizer files for each cluster
- **Datasets**: Paths to training/validation data
- **Benchmarks**: Paths to evaluation datasets (WikiText, LAMBADA)

See `user/example/cluster_paths.toml.example` for the complete template.

## Data Preparation

### Tokenize Dataset

```bash
apptainer exec --nv titan.sif \
    python titan_oellm/datasets/utils/preprocess_mmap_chunks.py \
    --input-folder /path/to/raw/data \
    --output-dir /path/to/chunks \
    --validate-only  # First validate
```

### Download Benchmarks

```bash
apptainer exec --nv titan.sif \
    python scripts/download_benchmarks.py \
    --tokenizer /path/to/tokenizer \
    --output-dir /path/to/benchmarks
```

## LR Schedulers

The framework supports multiple learning rate schedulers:

| Scheduler | Description |
|-----------|-------------|
| `wsd` | Warmup-Stable-Decay (TorchTitan default) |
| `wdd` | Warmup with gradual decay during stable phase |
| `cosine` | Cosine annealing with warmup |
| `universal` | 3-phase scheduler (warm -> main -> cooldown) |

### Universal Scheduler

The universal scheduler provides flexible 3-phase control:

```toml
[lr_scheduler]
scheduler_type = "universal"

# Phase 1: Warm (warmup or warmdown)
warm_steps = 200
warm_direction = "up"  # "up" for warmup, "down" for warmdown
warm_type = "linear"

# Phase 2: Main (stable or decaying)
main_decay_type = "const"  # or "linear", "cosine", "sqrt"
main_decay_ratio = 0.2

# Phase 3: Cooldown
cooldown_steps = 2000
cooldown_type = "cosine"
```



## Development

### Update TorchTitan Version

To update the TorchTitan submodule to a newer version:

```bash
# Navigate to the torchtitan submodule
cd torchtitan

# Fetch latest tags and branches
git fetch --all --tags

# Checkout desired version (e.g., v0.3.0)
git checkout v0.2.1

# Verify the version
git describe --tags

# Go back to project root
cd ..

# Commit the submodule update
git add torchtitan
git commit -m "Update torchtitan to v0.3.0"
```

After updating TorchTitan, rebuild the container:

```bash
# Update container definition if needed (titan_0.3.0.def)
apptainer build --fakeroot titan_juwels_0.2.1.sif titan_0.2.1.def
```


## Troubleshooting

### TITAN_USER not set
```
Error: TITAN_USER environment variable not set.
```
Solution: `export TITAN_USER=your_username`

### Configuration file not found
```
Configuration file not found: .../user/<username>/cluster_paths.toml
```
Solution: Create your config from the example template:
```bash
cp user/example/cluster_paths.toml.example user/$TITAN_USER/cluster_paths.toml
```

### Submodule not initialized
```
ModuleNotFoundError: No module named 'torchtitan'
```
Solution: `git submodule update --init --recursive`
