# Titan-OELLM

A TorchTitan-based framework for training large language models on HPC systems.

## Features

- **Custom Model Architectures**: GPT-Plus (with QKNormPlus) and Qwen3-Custom
- **HPC Optimization**: SLURM scripts for JUWELS, Jupiter, and Capella clusters
- **Flexible Configuration**: TOML-based configs with cluster-specific path resolution
- **Validation During Training**: Comprehensive validation with TensorBoard integration
- **Memory-Mapped Datasets**: Efficient data loading with chunking support

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive <repo-url>
cd titan-oellm

# Load and verify TorchTitan submodule
git submodule update --init --recursive
cd torchtitan && git describe --tags && cd ..
# Should output: v0.2.0
```

### 2. Set Up User Configuration

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

### 3. Build Container

```bash
export APPTAINER_CACHEDIR=/path/to/your/cache
export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp
apptainer build --fakeroot titan_CLUSTER_0.2.0.sif titan_0.2.0.def

```

Test container:
```bash
apptainer shell --nv titan_CLUSTER_0.2.0.sif
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4. Run Training

```bash
# Set your user (REQUIRED)
export TITAN_USER=your_username

# Submit training job
sbatch slurm_juwels.sh

# Or with custom dataset/config
DATASET=fineweb_edu CONFIG=qwen3_custom.toml sbatch slurm_juwels.sh
```

## Models

| Model | Description | Config |
|-------|-------------|--------|
| **gpt_plus** | GPT with QKNormPlus attention normalization and RoPE scaling | `base_plus.toml` |
| **qwen3_custom** | Qwen3 custom implementation with MoE support | `qwen3_custom.toml` |

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TITAN_USER` | Your username for user-specific configs | Yes |
| `DATASET` | Dataset name from cluster_paths.toml | No (default: slimpajama_627b) |
| `TOKENIZER` | Tokenizer name from cluster_paths.toml | No (default: neox) |
| `CONFIG` | Base config file | No (default: base_plus.toml) |
| `CLUSTER` | Override cluster detection | No (auto-detected) |

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
├── slurm_*.sh             # Cluster-specific job scripts
└── torchtitan/            # TorchTitan submodule (v0.2.0)
```

### User Configuration

Each user needs their own `user/<username>/cluster_paths.toml` with:

- **Cluster paths**: Output directories, cache locations
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
