---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #e6e6e6
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #00d2ff;
  }
  h2 {
    color: #7b68ee;
  }
  h3 {
    color: #48dbfb;
  }
  code {
    background-color: #16213e;
    color: #0ff;
  }
  pre {
    background-color: #16213e !important;
  }
  a {
    color: #48dbfb;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background-color: #16213e;
    color: #00d2ff;
  }
  td {
    background-color: #0f3460;
  }
  strong {
    color: #f39c12;
  }
  blockquote {
    border-left: 4px solid #7b68ee;
    background-color: #16213e;
    padding: 0.5em 1em;
  }
---

<!-- _class: lead -->

# Titan-OELLM

### A TorchTitan Wrapper for Scalable LLM Training on HPC Systems

---

## Agenda

1. **Motivation** -- Why a wrapper around TorchTitan?
2. **Architecture Overview** -- Repository structure & design
3. **Configuration System** -- TOML configs, cluster paths, environment
4. **Starting an Experiment** -- Local & SLURM workflows
5. **Key Components** -- Data loaders, validators, schedulers, logging
6. **Adding New Models** -- Step-by-step guide
7. **Adding New Features** -- Datasets, tokenizers, schedulers
8. **Live Demo / Q&A**

---

## What is TorchTitan?

**TorchTitan** (v0.2.1) is PyTorch's native platform for large-scale LLM training.

It provides:
- Multi-dimensional parallelism (FSDP2, Tensor Parallel, Pipeline Parallel, Context Parallel)
- Distributed checkpointing (DCP)
- Configuration management via TOML + tyro
- `torch.compile` integration
- Built-in models (Llama3, etc.)

> TorchTitan handles the *distributed training engine*.
> We need everything else around it for our HPC environments.

---

## Why a Wrapper?

### Problems TorchTitan alone doesn't solve for us:

| Challenge | Our Solution |
|-----------|-------------|
| Multiple HPC clusters (JUWELS, Jupiter, Capella, Leonardo) | `cluster_config.py` -- auto-detection & path resolution |
| Per-user dataset/tokenizer paths | `user/$USER/cluster_paths.toml` |
| Custom model architectures | Model registration system (`TrainSpec`) |
| Validation during training | Multi-metric validator (Perplexity, WikiText, LAMBADA) |
| Training diagnostics | Parameter & gradient logging to TensorBoard |
| Flexible LR schedules | Universal 3-phase scheduler |
| Efficient data loading for large corpora | MMap, Chunked, Deterministic Packed dataloaders |
| Containerized execution | Apptainer/Singularity integration |

---

## Architecture Overview

```
titan-oellm/
 ├── torchtitan/              <-- Git submodule (upstream, untouched)
 ├── titan_oellm/             <-- Our extensions
 │    ├── cluster_config.py        Cluster auto-detection & paths
 │    ├── configs/                 Extended JobConfig
 │    ├── components/              Validator, LR scheduler, param logger
 │    ├── datasets/                Dataloaders, tokenizer, collator
 │    └── models/                  Custom model implementations
 ├── titan_train.py           <-- Entry point (config monkey-patch)
 ├── submit_job.sh            <-- Unified job submission
 ├── slurm/                   <-- Per-cluster SLURM scripts
 └── user/                    <-- Per-user configurations
```

**Key principle**: TorchTitan is used **as-is** (submodule). All customization lives in `titan_oellm/`.

---

## Design Philosophy

```
                    +----------------------------+
                    |       TorchTitan Core      |
                    |  (Parallelism, DCP, Loop)   |
                    +-------------+--------------+
                                  |
                    +-------------+--------------+
                    |      titan_oellm Layer     |
                    |  Models, Data, Validation,  |
                    |  Config, Cluster, Logging   |
                    +-------------+--------------+
                                  |
          +-----------+-----------+-----------+
          |           |           |           |
     cluster_config  models    datasets   components
     (path mgmt)   (Qwen3)   (MMap etc) (validator,
                                          scheduler,
                                          param log)
```

All components are **pluggable** via the `TrainSpec` registration pattern.

---

## Configuration System

### Three layers of configuration:

1. **Model training config** (TOML) -- model architecture, optimizer, parallelism
   `titan_oellm/models/qwen3_custom/train_configs/qwen3_custom.toml`

2. **User cluster paths** (TOML) -- dataset/tokenizer paths per cluster
   `user/$USER/cluster_paths.toml`

3. **Environment variables** -- runtime overrides
   `TITAN_USER`, `CLUSTER`, `DATASET`, `TOKENIZER`, `CONFIG`, `NPROC`

> Configs are **cluster-independent**. Paths are injected at runtime by `cluster_config.py`.

---

## Training Config (TOML)

```toml
[model]
name = "qwen3_custom"
flavor = "1.7B"              # Model size variant

[training]
steps = 100000
seq_len = 2048
local_batch_size = 4
mixed_precision_param = "bfloat16"

[optimizer]
name = "AdamW"
lr = 3e-4

[lr_scheduler]
scheduler_type = "universal"
warm_steps = 1000
warm_type = "linear"
main_decay_type = "cosine"
cooldown_steps = 500

[parallelism]
dp_shard = -1                # Auto
tp = 1
```

---

## User Cluster Paths

```toml
# user/$USER/cluster_paths.toml

["cluster.juwels"]
output_dir = "/p/scratch/project/user/titan_output"
cache_base = "/p/scratch/project/user/cache"

["tokenizer.neox.juwels"]
path = "/p/scratch/project/tokenizers/neox"

["dataset.slimpajama_627b.neox.juwels"]
train_prefix = "/p/scratch/project/data/slimpajama/train"
validation_prefix = "/p/scratch/project/data/slimpajama/val"
dataloader = "ChunkedMMapDataset"
min_doc_len = 128
```

**Lookup pattern**: `dataset.{name}.{tokenizer}.{cluster}`

---

## Path Resolution Flow

```
  Environment Variables          cluster_paths.toml
  ┌──────────────────┐          ┌──────────────────┐
  │ DATASET=slimpaj  │          │ dataset paths    │
  │ TOKENIZER=neox   │────┐     │ tokenizer paths  │
  │ CLUSTER=juwels   │    │     │ cluster settings │
  └──────────────────┘    │     └────────┬─────────┘
                          │              │
                    ┌─────▼──────────────▼─────┐
                    │    cluster_config.py      │
                    │    get_cli_args()         │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │  --job.config_file=...    │
                    │  --data.data_prefix=...   │
                    │  --data.chunks_dir=...    │
                    │  --validation.data_prefix │
                    │  --benchmarks.wikitext2   │
                    └──────────────────────────┘
                         CLI args for torchrun
```

---

## Starting an Experiment

### Option 1: Local (development/testing)

```bash
TITAN_USER=joerg \
DATASET=test_dataset \
TOKENIZER=neox \
CONFIG=user/joerg/configs/debug.toml \
NPROC=1 \
  bash submit_job.sh --local \
    --model.flavor=debugmodel \
    --training.seq_len=512
```

### Option 2: SLURM (production)

```bash
TITAN_USER=joerg \
DATASET=slimpajama_627b \
  bash submit_job.sh --nodes=4 -- \
    --model.flavor=1.7B \
    --training.steps=100000
```

---

## Experiment Launch Flow

```
  submit_job.sh
       |
       +-- [--local mode]
       |      |
       |      +-- Load cluster_config.py
       |      +-- Resolve paths (get_cli_args)
       |      +-- Set env vars (get_env_exports)
       |      +-- apptainer exec ... torchrun -m torchtitan.train $CLI_ARGS
       |
       +-- [SLURM mode]
              |
              +-- Create .venv_submit (no torch needed)
              +-- Detect cluster from hostname
              +-- Find container (titan_${CLUSTER}_0.2.1.sif)
              +-- sbatch slurm/$CLUSTER.sh $TRAINING_ARGS
                     |
                     +-- Load modules (CUDA, NCCL, ...)
                     +-- Load cluster_config.py (inside container)
                     +-- Set NCCL env vars
                     +-- torchrun --nnodes=$NODES ... -m torchtitan.train
```

---

## Key Components: Data Loaders

### Three dataloader types:

| Type | Use Case | Key Feature |
|------|----------|-------------|
| **MMapDataset** | Single large binary file | Memory-mapped random access |
| **ChunkedMMapDataset** | Pre-chunked data | Deterministic iteration, validation split |
| **DeterministicPackedDataset** | Document packing | Fixed-length sequences, reproducible |

```
Raw Text  -->  Tokenize  -->  .bin/.idx files  -->  MMap/Chunked Loader
                                                          │
                                                    collate_function()
                                                          │
                                                    (input_ids, labels)
```

Configured via `[data]` section and `cluster_paths.toml`.

---

## Why Chunks? The Problem with Single MMap Files

### Scaling issues with one big .bin file:

- **Shuffling**: A single file stores documents sequentially. True random access across a 2TB file causes **random I/O thrashing** on parallel file systems (GPFS, Lustre).
- **Multi-node data loading**: All ranks read from one file = I/O contention.
- **Validation split**: Hard to hold out data without a second copy of the file.
- **Worker independence**: With a single file, different `dp_world_size` or `num_workers` settings change which data each rank sees -- **not reproducible**.

### What chunks solve:

- **Pre-shuffled chunks** (e.g. 256 chunks of ~8GB each) -- each chunk is internally shuffled at creation time, so sequential reads within a chunk are already random.
- **Round-robin assignment** -- each DP rank gets its own subset of chunks, minimizing I/O contention.
- **Built-in validation split** -- reserve first N docs per chunk for validation, rest for training. No separate files needed.
- **Deterministic across node counts** -- same seed + same chunks = same data order.

---

## preprocess_mmap_chunks: Creating Chunks

`titan_oellm/datasets/utils/preprocess_mmap_chunks.py`

Converts a single large .bin/.idx dataset into shuffled chunks:

```
Input:  train.bin (2TB) + train.idx
                |
    preprocess_mmap_chunks.py
    (parallel workers + async flush)
                |
Output: chunks_dir/
          chunk_0000.bin + chunk_0000.idx
          chunk_0001.bin + chunk_0001.idx
          ...
          chunk_0255.bin + chunk_0255.idx
```

### Key features:
- **Multi-process**: `ProcessPoolExecutor` for parallel reading
- **Async disk writes**: Separate writer processes so readers never block
- **Random assignment**: Each document is randomly assigned to a target chunk
- **Per-chunk shuffle**: Each chunk is shuffled independently after assignment
- **Validation**: Automatic integrity check after creation
- **Cleanup**: Temp files deleted only after validation passes

---

## ChunkedMMapDataset: How It Works

```
                    All chunks (sorted, deterministic)
                    [chunk_00, chunk_01, chunk_02, ..., chunk_N]
                              |
                    Seed-based permutation per epoch
                              |
                    Round-robin assignment to DP ranks
                              |
          Rank 0              Rank 1              Rank 2
          [chunk_02,          [chunk_07,          [chunk_00,
           chunk_05,           chunk_01,           chunk_09,
           chunk_11, ...]      chunk_04, ...]      chunk_06, ...]
              |                    |                    |
          Sequential read     Sequential read     Sequential read
          within each chunk   within each chunk   within each chunk
```

- Each rank reads its chunks **sequentially** (fast I/O, no random seeks)
- Chunks are **pre-shuffled** at creation time, so sequential = random
- New epoch = re-shuffle chunk order (seed + epoch_counter)
- **Validation split**: `use_only_first_n_per_chunk` / `exclude_first_n_per_chunk`

---

## DeterministicPackedDataset: Deep Dive

### The problem it solves:
Documents have **variable length**. Naive approaches either pad (wasted compute) or use per-rank streaming (non-deterministic across node counts).

### Greedy packing approach:
```
Doc1 (500 tok) | Doc2 (1200 tok) | Doc3 (800 tok) | Doc4 (300 tok) | ...
                            |
              Treat as one continuous token stream
                            |
         [--- seq_len+1 ---][--- seq_len+1 ---][--- seq_len+1 ---]
              Sequence 0         Sequence 1         Sequence 2
```

Documents are concatenated (with optional EOS separators) into a virtual token stream. Fixed-length sequences are cut at regular positions -- **no stored index needed**, just binary search on cumulative token counts.

---

## DeterministicPackedDataset: Batch Diversity

### Strided assignment ensures every batch samples across the whole dataset:

```
  Epoch token stream (all chunks, permuted):
  [===========================================================]
   ^           ^           ^           ^           ^
   lane 0     lane 1     lane 2     lane 3     lane 4
   (step 0)   (step 0)   (step 0)   (step 0)   (step 0)

  Step 0 batch = {lane_0[0], lane_1[0], lane_2[0], ...}
  Step 1 batch = {lane_0[1], lane_1[1], lane_2[1], ...}
```

- Each lane advances sequentially through its region (fast I/O)
- But batch spans the **entire dataset** (diversity)
- Batch composition depends only on `global_batch_size`, **not** on `dp_world_size`
- **Checkpoint = 1 integer** (`global_sequence_id`) -- instant resume, no fast-forward

### Scales to: 1-10T tokens, 100-4000 GPUs, global_batch_size up to 16M.

---

## Key Components: Validator

Multi-metric validation during training:

- **Perplexity** -- cross-entropy loss on validation set
- **WikiText-2 / WikiText-103** -- standard LM benchmarks
- **LAMBADA** -- last-word prediction accuracy
- **Spearman Correlation** -- rank correlation metric

```toml
[validation]
enable = true
freq = 1000          # Validate every N steps
eval_mode = "concatenated"  # or "document"

[benchmarks]
wikitext2_path = "..."    # Injected by cluster_config
lambada_path = "..."
```

Supports **multiple validation datasets** per training run.

---

## Key Components: Universal LR Scheduler

Three-phase learning rate schedule:

```
   LR
    ^
    |        Phase 1       Phase 2          Phase 3
    |       (Warm)         (Main)          (Cooldown)
    |
    |          /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    |         /                       \
    |        /    linear/cosine/       \  cosine/linear
    |       /     constant decay        \  decay
    |      /                             \
    |     /                               \___________
    |    /                                     lr_min
    +----+--------+-------------------+--------+----->
         0    warm_steps         total-cooldown    steps
```

Supports: `linear`, `cosine`, `sqrt`, `exp` decay types per phase.

---

## Key Components: Parameter Logger

Statistical insights into training dynamics:

```toml
[parameter_logging]
enabled = true
log_interval = 1000
log_parameters = true       # Weight statistics (max, min, norm, std)
log_gradients = true        # Gradient statistics
log_optimizer_states = true # Adam momentum/variance statistics
```

- Pattern-based filtering (include/exclude layers)
- All stats logged to **TensorBoard**
- Helps diagnose: gradient explosion, dead layers, optimizer issues

---

## Model Registration: TrainSpec Pattern

Every model registers a `TrainSpec` -- a single object that wires everything together:

```python
TrainSpec(
    model_cls       = Qwen3Model,          # The model class
    model_args      = qwen3_custom_configs, # Dict of size variants
    parallelize_fn  = parallelize_qwen3,   # FSDP/TP/AC setup
    pipelining_fn   = ...,                 # Pipeline parallel
    build_optimizers_fn     = ...,
    build_lr_schedulers_fn  = ...,
    build_dataloader_fn     = build_sci_dataloader,
    build_tokenizer_fn      = build_sci_hf_tokenizer,
    build_loss_fn           = ...,
    build_validator_fn      = build_validator,
    build_metrics_processor_fn = ...,
    state_dict_adapter      = Qwen3StateDictAdapter,
)

register_train_spec("qwen3_custom", spec)
```

---

## Current Model: Qwen3-Custom

### Available flavors (Qwen3 official sizes):

| Flavor | dim | layers | heads | kv_heads |
|--------|-----|--------|-------|----------|
| debugmodel | 128 | 2 | 2 | 2 |
| 0.5B | 896 | 24 | 14 | 2 |
| 0.6B | 1024 | 28 | 16 | 8 |
| 1.7B | 1536 | 28 | 12 | 2 |
| 4B | 2560 | 36 | 20 | 4 |
| 8B | 3584 | 36 | 28 | 4 |
| 14B | 5120 | 40 | 40 | 8 |
| 32B | 5120 | 64 | 40 | 8 |

Plus: **125M**, **125M768**, **130Msci**, **1.7Bsci** (custom research sizes) and **MoE variants** (debugmodel_moe, 600M-A60M).

Features: QK-norm, configurable RoPE theta, HF weight loading, depth init, weight tying.

---

## Adding a New Model -- Step by Step

### 1. Create the directory structure

```
titan_oellm/models/my_model/
├── __init__.py              # TrainSpec registration
├── model/
│   ├── args.py              # Model hyperparameters
│   ├── model.py             # nn.Module implementation
│   └── state_dict_adapter.py  # (optional) HF weight conversion
├── infra/
│   └── parallelize.py       # Parallelism setup
└── train_configs/
    └── my_model.toml        # Default training config
```

---

## Adding a New Model -- Step by Step (cont.)

### 2. Define model arguments

```python
# model/args.py
@dataclass
class MyModelArgs(BaseModelArgs):
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    vocab_size: int = 50432
```

### 3. Implement the model

```python
# model/model.py
class MyModel(nn.Module):
    def __init__(self, model_args: MyModelArgs):
        super().__init__()
        # Build layers ...

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Forward pass ...
        return output
```

---

## Adding a New Model -- Step by Step (cont.)

### 4. Implement parallelization

```python
# infra/parallelize.py
def parallelize_my_model(model, parallel_dims, job_config):
    if parallel_dims.tp_enabled:
        apply_tp(model, ...)
    if parallel_dims.dp_shard_enabled:
        apply_fsdp(model, ...)
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, ...)
```

### 5. Register the TrainSpec

```python
# __init__.py
my_model_configs = {"base": MyModelArgs(...), "large": MyModelArgs(...)}

spec = TrainSpec(model_cls=MyModel, model_args=my_model_configs, ...)
register_train_spec("my_model", spec)
```

### 6. Add import in `titan_oellm/models/__init__.py`

```python
from . import my_model   # Auto-registers on import
```

---

## Adding a New Dataset

### 1. Tokenize your data

```bash
python titan_oellm/scripts/convert_hf_to_mmap.py \
    --input your_data/ \
    --output /scratch/data/my_dataset/ \
    --tokenizer /path/to/tokenizer
```

### 2. Register in `user/$USER/cluster_paths.toml`

```toml
["dataset.my_dataset.neox.juwels"]
train_prefix = "/scratch/data/my_dataset/train"
validation_prefix = "/scratch/data/my_dataset/val"
dataloader = "MMapDataset"
min_doc_len = 64
```

### 3. Use it

```bash
DATASET=my_dataset bash submit_job.sh --local
```

---

## Adding a New Tokenizer

### 1. Register in cluster paths

```toml
["tokenizer.my_tokenizer.juwels"]
path = "/scratch/tokenizers/my_tokenizer"
```

### 2. Tokenize data with the new tokenizer

Create dataset entries matching the new tokenizer:

```toml
["dataset.my_data.my_tokenizer.juwels"]
train_prefix = "/scratch/data/my_data_mytok/train"
...
```

### 3. Use it

```bash
DATASET=my_data TOKENIZER=my_tokenizer bash submit_job.sh --local
```

---

## Multi-Cluster Support

### Supported HPC Systems

| Cluster | Location | Detection Pattern | Container |
|---------|----------|-------------------|-----------|
| **JUWELS** | FZ Juelich | `jwlogin*`, `jwc*`, `juwels` | `titan_juwels_0.2.1.sif` |
| **Jupiter** | FZ Juelich | `jupiter*`, `jrc*` | `titan_jupiter_0.2.1.sif` |
| **Capella** | PSNC | `c` + digit, `capella` | `titan_capella_0.2.1.sif` |
| **Leonardo** | CINECA | `leonardo` | `titan_leonardo_0.2.1.sif` |
| **Local** | Dev machine | fallback | local container |

Cluster is **auto-detected** from hostname. Override with `CLUSTER=...`.

Same training config runs everywhere -- only paths differ.

---

## Execution Environment

```
  Login Node                          Compute Node
  ┌──────────────────┐               ┌──────────────────────────────┐
  │                  │               │  Apptainer Container         │
  │  submit_job.sh   │               │  ┌────────────────────────┐  │
  │  .venv_submit    │   sbatch      │  │  torchrun              │  │
  │  (no torch!)     │ ──────────>   │  │  torchtitan.train      │  │
  │  cluster_config  │               │  │  titan_oellm (models,  │  │
  │                  │               │  │    data, components)   │  │
  └──────────────────┘               │  └────────────────────────┘  │
                                     │  GPU 0  GPU 1  GPU 2  GPU 3  │
                                     └──────────────────────────────┘
```

- Login node: lightweight venv (no PyTorch) for config resolution
- Compute node: full environment inside Apptainer container
- All paths bind-mounted into container

---

## Typical Workflow

```
1.  Clone repo, set up user config
        user/$USER/cluster_paths.toml

2.  Choose or create training config
        user/$USER/configs/my_experiment.toml

3.  (Optional) Tokenize data & download benchmarks
        python scripts/convert_hf_to_mmap.py
        python scripts/download_benchmarks.py

4.  Submit experiment
        TITAN_USER=$USER DATASET=my_data \
          bash submit_job.sh --nodes=8 -- \
            --model.flavor=4B --training.steps=50000

5.  Monitor training
        tensorboard --logdir outputs/
```

---

## Summary

### What Titan-OELLM gives you:

- **Seamless multi-cluster support** -- same config, any HPC system
- **Pluggable architecture** -- swap models, dataloaders, schedulers
- **Production-ready training** -- validation, logging, checkpointing
- **Easy onboarding** -- user directory, environment variables, TOML configs
- **Extensibility** -- add a model in ~5 files, a dataset in ~3 lines

### What TorchTitan gives us (under the hood):

- FSDP2, Tensor Parallel, Pipeline Parallel
- Distributed checkpointing
- Training loop, optimizer, compilation
- All the hard distributed systems work

---

<!-- _class: lead -->

# Questions?

### Key resources:
- `README.md` -- Quick start & overview
- `titan_oellm/configs/README.md` -- Config system docs
- `titan_oellm/models/qwen3_custom/README.md` -- Model docs
- `user/example/` -- Template configurations

---

<!-- _class: lead -->

# Appendix

---

## Appendix A: Full Config Sections

| TOML Section | Purpose |
|-------------|---------|
| `[job]` | Output folder, config module |
| `[model]` | Architecture name, flavor, vocab size |
| `[training]` | Steps, batch size, seq_len, precision |
| `[optimizer]` | AdamW params (lr, betas, weight_decay) |
| `[lr_scheduler]` | Scheduler type, phase config |
| `[parallelism]` | DP, TP, PP, CP degrees |
| `[data]` | Dataloader type, data paths |
| `[validation]` | Enable, frequency, metrics |
| `[benchmarks]` | WikiText, LAMBADA paths |
| `[parameter_logging]` | Stats logging config |
| `[checkpoint]` | Enable, interval, HF loading |
| `[compile]` | torch.compile settings |
| `[activation_checkpoint]` | AC mode (full/selective) |

---

## Appendix B: Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TITAN_USER` | *(required)* | Your username |
| `CLUSTER` | auto-detected | Target cluster |
| `DATASET` | `test_dataset` (local) | Dataset name |
| `TOKENIZER` | `neox` | Tokenizer name |
| `CONFIG` | `debug.toml` (local) | Config file |
| `NPROC` | `1` | GPUs (local mode) |
| `OUTPUT_DIR` | from cluster config | Output base path |

---

## Appendix C: Adding a New LR Scheduler

1. Implement scheduler in `titan_oellm/components/`:

```python
# components/my_scheduler.py
class MyScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, ...):
        ...
    def get_lr(self):
        ...
```

2. Add builder function and register in the TrainSpec's `build_lr_schedulers_fn`.

3. Add config fields to `oellm_job_config.py` if needed.

---

## Appendix D: HuggingFace Weight Loading

Load pretrained HF checkpoints directly:

```toml
[checkpoint]
enable_checkpoint = true
load_hf_model_weights_only = true
hf_model_weights_path = "/path/to/hf/model"
```

The `StateDictAdapter` handles:
- Weight name mapping (HF naming -> TorchTitan naming)
- Shape conversions (e.g., fused QKV -> separate Q, K, V)
- Skipping incompatible layers

Supports both **fine-tuning** and **continual pretraining**.
