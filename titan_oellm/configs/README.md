# Titan-Sci Configuration System

> **Deprecated location.** `cluster_paths.toml` no longer lives in this directory.
> Cluster paths are now read from a single, local (gitignored) `user/cluster_paths.toml`
> at the repo root — copy `user/example/cluster_paths.toml.example` to get started.
> The `TITAN_USER` env var and per-user `user/<name>/` folders have been removed.
> See the repository `README.md` ("User Configuration") for the current workflow.
> The rest of this document is kept for historical reference.

This directory contains the unified configuration system for Titan-Sci training across multiple HPC clusters.

## Overview

The configuration system is designed to:
- **Eliminate duplication**: One set of hyperparameters for all clusters
- **Easy dataset switching**: Change datasets via environment variables
- **Auto-detection**: Automatically detects cluster from hostname
- **Backwards compatible**: Old cluster-specific configs still work

## File Structure

```
configs/
├── cluster_paths.toml       # All cluster-specific paths (datasets & tokenizers)
├── base_norm.toml           # Unified config for norm_gpt model
├── base_plus.toml           # Unified config for gpt_plus model
├── local_default.toml       # For local development (kept as-is)
├── juwels_runtime.toml      # For debugging (kept as-is)
└── [legacy configs]         # Old cluster-specific configs (deprecated)
```

## Quick Start

### Default Usage
```bash
# Use default settings (juwels + slimpajama_627b + neox tokenizer + norm_gpt)
sbatch slurm_juwels.sh
```

### Change Cluster
```bash
# Use different cluster paths (e.g., for testing Jupiter paths on JUWELS)
CLUSTER=jupiter sbatch slurm_juwels.sh
```

### Change Dataset
```bash
# Use a different dataset (must be defined in cluster_paths.toml)
DATASET=fineweb_edu sbatch slurm_juwels.sh
```

### Change Tokenizer
```bash
# Use a different tokenizer (must be defined in cluster_paths.toml)
TOKENIZER=llama3 sbatch slurm_juwels.sh
```

### Change Model
```bash
# Use gpt_plus instead of norm_gpt
CONFIG=base_plus.toml sbatch slurm_juwels.sh
```

### Combine All
```bash
# Custom combination
CLUSTER=capella DATASET=cosmopedia TOKENIZER=llama3 CONFIG=base_plus.toml sbatch slurm_juwels.sh
```

### Override Hyperparameters
```bash
# Pass additional CLI arguments
sbatch slurm_juwels.sh --model.flavor=1B --training.steps=50000 --optimizer.lr=2e-4
```

## Adding a New Dataset

1. **Edit `cluster_paths.toml`** and add 3 sections (one per cluster):

```toml
["dataset.my_new_dataset.neox.juwels"]
train_prefix = "/p/scratch/.../my_dataset/train/merged"
train_chunks = "/p/scratch/.../my_dataset/train/chunks"
validation_prefix = "/p/data1/.../my_dataset/validation/merged"
dataloader = "ChunkedMMapDataset"
min_doc_len = 64

["dataset.my_new_dataset.neox.jupiter"]
train_prefix = "/e/project1/.../my_dataset/train/merged"
# ... same structure

["dataset.my_new_dataset.neox.capella"]
train_prefix = "/data/horse/.../my_dataset/train/merged"
# ... same structure
```

2. **Use it**:
```bash
DATASET=my_new_dataset sbatch slurm_juwels.sh
```

## Adding a New Tokenizer

1. **Edit `cluster_paths.toml`** and add 3 sections:

```toml
["tokenizer.my_tokenizer.juwels"]
path = "/p/project1/.../tokenizer/my_tokenizer"

["tokenizer.my_tokenizer.jupiter"]
path = "/e/project1/.../tokenizer/my_tokenizer"

["tokenizer.my_tokenizer.capella"]
path = "/data/horse/.../tokenizer/my_tokenizer"
```

2. **Use it**:
```bash
TOKENIZER=my_tokenizer sbatch slurm_juwels.sh
```

## Configuration Files

### `cluster_paths.toml`
Central registry of all cluster-specific paths. Format:
```toml
["tokenizer.{name}.{cluster}"]
path = "/absolute/path/to/tokenizer"

["dataset.{name}.{tokenizer}.{cluster}"]
train_prefix = "/absolute/path/to/train/data"
train_chunks = "/absolute/path/to/train/chunks"
validation_prefix = "/absolute/path/to/validation/data"
dataloader = "ChunkedMMapDataset"  # or "MMapDataset"
min_doc_len = 64
```

**Note:** Dataset keys explicitly include the tokenizer name since datasets are pre-tokenized.

### `base_norm.toml` / `base_plus.toml`
Unified model configurations containing:
- Model architecture (flavor, layers, activation, etc.)
- Training hyperparameters (lr, batch size, steps, etc.)
- Optimizer settings
- Parallelism configuration
- **NO cluster-specific paths** (injected at runtime)

## Python API

```python
from titan_oellm.cluster_config import get_paths, get_cli_args, list_available

# Get all paths for a dataset/tokenizer combo
paths = get_paths('slimpajama_627b', 'neox')
print(paths['tokenizer_path'])
print(paths['data_prefix'])

# Generate CLI arguments
args = get_cli_args('slimpajama_627b', 'neox')
# Returns: "--model.tokenizer_path=... --data.data_prefix=... ..."

# List available configurations
list_available()
# Output:
#   Available configurations:
#     Clusters: capella, jupiter, juwels
#     Datasets: slimpajama_627b
#     Tokenizers: neox
```

## Command-Line Interface

```bash
# List available configs
python titan_oellm/cluster_config.py list

# Detect current cluster
python titan_oellm/cluster_config.py detect

# Get CLI args for specific config
python titan_oellm/cluster_config.py slimpajama_627b neox juwels
```

## Cluster Configuration

The cluster is configured via the `CLUSTER` environment variable in the SLURM script:
- **JUWELS**: `CLUSTER=juwels` (default in slurm_juwels.sh)
- **Jupiter**: `CLUSTER=jupiter` (default in slurm_jupiter.sh)
- **Capella**: `CLUSTER=capella` (default in slurm_capella.sh)

This is more explicit and predictable than auto-detection from hostname.

**Note:** The `detect_cluster()` function in `cluster_config.py` is still available for interactive use, but SLURM scripts use explicit configuration.

## Migration from Old Configs

Old cluster-specific configs (`juwels_norm.toml`, etc.) still work but are deprecated. To migrate:

1. Update your SLURM script to use the new system (see `slurm_juwels.sh`)
2. Use `base_norm.toml` or `base_plus.toml` instead of cluster-specific configs
3. Dataset/tokenizer paths are automatically resolved

## Benefits

✅ **18 configs → 2 base configs** + 1 path file
✅ **Single source of truth** for hyperparameters
✅ **Easy dataset switching** via environment variables
✅ **Automatic cluster detection**
✅ **Simple to add new datasets** (3 TOML sections)
✅ **Backwards compatible** during migration
✅ **Type-safe** TOML format with validation

## Troubleshooting

**Error: "Unknown cluster for hostname"**
- The cluster couldn't be auto-detected
- Solution: Manually specify cluster in SLURM script or add hostname pattern to `cluster_config.py`

**Error: "Dataset 'X' not found for cluster 'Y'"**
- The dataset isn't configured for that cluster
- Solution: Add the dataset to `cluster_paths.toml` for all 3 clusters

**Error: "Tokenizer 'X' not found for cluster 'Y'"**
- The tokenizer isn't configured for that cluster
- Solution: Add the tokenizer to `cluster_paths.toml` for all 3 clusters

## Support

For issues or questions:
1. Check this README
2. Run `python titan_oellm/cluster_config.py list` to see available configs
3. Review `cluster_paths.toml` to verify paths are correct
