# Qwen3 Custom Model for Titan-Sci

This is a custom Qwen3 model implementation integrated with the titan-sci training infrastructure. It supports:

- **Loading pretrained Qwen3 weights from HuggingFace** for fine-tuning or continual pretraining
- **Full titan-sci integration**: sci_dataloader, universal LR schedulers, validation, parameter logging
- **Architecture customization**: Modify the Qwen3 architecture while loading compatible weights
- **All Qwen3 model sizes**: 0.5B, 0.6B, 1.7B, 4B, 8B, 14B, 32B (dense and MoE variants)

## Quick Start

### 1. Basic Training from Scratch

```bash
python train.py --job.config_file=configs/qwen3_custom.toml
```

This will train a Qwen3 0.6B model from random initialization using the titan-sci infrastructure.

### 2. Fine-tuning from HuggingFace Checkpoint

See the [Loading Pretrained Weights](#loading-pretrained-weights-from-huggingface) section below.

## Model Configurations

Available model sizes (defined in `__init__.py:qwen3_custom_configs`):

| Model Size | Dim  | Layers | Heads | KV Heads | Hidden Dim | Parameters |
|------------|------|--------|-------|----------|------------|------------|
| debugmodel | 512  | 8      | 8     | 4        | 1536       | ~50M       |
| 0.5B       | 896  | 24     | 14    | 2        | 4864       | ~0.5B      |
| 0.6B       | 1024 | 28     | 16    | 8        | 3072       | ~0.6B      |
| 1.7B       | 1536 | 28     | 12    | 2        | 8960       | ~1.7B      |
| 4B         | 2560 | 36     | 20    | 4        | 13824      | ~4B        |
| 8B         | 3584 | 36     | 28    | 4        | 18944      | ~8B        |
| 14B        | 5120 | 40     | 40    | 8        | 13824      | ~14B       |
| 32B        | 5120 | 64     | 40    | 8        | 27648      | ~32B       |

Select a model size by setting `model.flavor` in your config:

```toml
[model]
name = "qwen3_custom"
flavor = "0.6B"  # or "1.7B", "4B", etc.
```

## Loading Pretrained Weights from HuggingFace

There are two methods to load pretrained Qwen3 weights from HuggingFace:

### Method 1: Direct Loading During Training (Recommended)

This method loads HuggingFace safetensors directly at training initialization, automatically converting the state dict format.

#### Step 1: Download HuggingFace Model

Download the pretrained Qwen3 model and tokenizer from HuggingFace:

```bash
# Download using the torchtitan download script
python torchtitan/scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-0.6B \
    --assets model tokenizer

# Or download manually using huggingface-cli
huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir ./assets/hf/Qwen3-0.6B
```

**Available Qwen3 Models on HuggingFace:**
- `Qwen/Qwen3-0.5B`, `Qwen/Qwen3-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-0.6B-Instruct`
- `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-1.7B-Instruct`
- `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Instruct`
- `Qwen/Qwen3-8B`, `Qwen/Qwen3-8B-Instruct`
- `Qwen/Qwen3-14B`, `Qwen/Qwen3-14B-Instruct`
- `Qwen/Qwen3-32B`, `Qwen/Qwen3-32B-Instruct`

#### Step 2: Configure Training to Load HF Checkpoint

Edit your config file (e.g., `configs/qwen3_custom.toml`) and enable HF checkpoint loading:

```toml
[model]
name = "qwen3_custom"
flavor = "0.6B"  # Must match the HF model size
vocab_size = 151936  # Qwen3 vocab size
# Set tokenizer_path to the HF model directory
tokenizer_path = "./assets/hf/Qwen3-0.6B"

[checkpoint]
enable = false  # Not saving checkpoints initially
# Enable HF loading
initial_load_in_hf = true
initial_load_model_only = true  # Only load model weights, not optimizer state
initial_load_path = "./assets/hf/Qwen3-0.6B"  # Path to HF checkpoint directory
```

**Important Configuration Notes:**

1. **`model.flavor` must match the HF model size**
   - For `Qwen/Qwen3-0.6B`, use `flavor = "0.6B"`
   - For `Qwen/Qwen3-1.7B`, use `flavor = "1.7B"`
   - etc.

2. **`initial_load_model_only = true`** ensures only model weights are loaded (not optimizer state)
   - Set to `false` if resuming from a titan-sci checkpoint with optimizer state

3. **`initial_load_path`** can be:
   - Absolute path: `/path/to/Qwen3-0.6B`
   - Relative path: `./assets/hf/Qwen3-0.6B`
   - HuggingFace model ID: `Qwen/Qwen3-0.6B` (requires `transformers` and internet)

#### Step 3: Start Training

```bash
python train.py --job.config_file=configs/qwen3_custom.toml
```

The training script will:
1. Detect the `initial_load_in_hf=true` flag
2. Load weights from the HF checkpoint using `Qwen3StateDictAdapter`
3. Convert the HF state dict format to titan-sci format
4. Initialize the model with pretrained weights
5. Start training (fine-tuning or continual pretraining)

### Method 2: Offline Conversion (Advanced)

This method pre-converts the HuggingFace checkpoint to TorchTitan Distributed Checkpoint (DCP) format before training.

#### Step 1: Download HuggingFace Model

Same as Method 1 Step 1.

#### Step 2: Convert HF Checkpoint to DCP Format

```bash
python torchtitan/scripts/checkpoint_conversion/convert_from_hf.py \
    ./assets/hf/Qwen3-0.6B \
    ./checkpoints/qwen3_0.6B_dcp \
    --model_name qwen3 \
    --model_flavor 0.6B
```

This will create a TorchTitan-format checkpoint in `./checkpoints/qwen3_0.6B_dcp/`.

#### Step 3: Configure Training to Load DCP Checkpoint

Edit your config file:

```toml
[model]
name = "qwen3_custom"
flavor = "0.6B"
vocab_size = 151936
tokenizer_path = "./assets/hf/Qwen3-0.6B"

[checkpoint]
enable = true
folder = "./checkpoints/qwen3_0.6B_dcp"
# No initial_load_* flags needed - checkpoint folder is auto-detected
```

#### Step 4: Start Training

```bash
python train.py --job.config_file=configs/qwen3_custom.toml
```

The training script will automatically detect and load the checkpoint from the specified folder.

## Fine-tuning vs. Continual Pretraining

### Fine-tuning (Supervised)

Load a pretrained model and train on a supervised dataset:

```toml
[training]
dataset = "sci_dataset"  # Your supervised dataset
local_batch_size = 8
seq_len = 2048
steps = 10000

[optimizer]
lr = 1e-5  # Lower LR for fine-tuning
weight_decay = 0.1

[lr_scheduler]
scheduler_type = "cosine"
warm_steps = 500
main_decay_type = "cosine"
main_decay_ratio = 0.1
```

### Continual Pretraining (Unsupervised)

Load a pretrained model and continue pretraining on more data:

```toml
[training]
dataset = "sci_dataset"  # Your pretraining dataset
local_batch_size = 8
seq_len = 4096  # Longer sequences for pretraining
steps = 100000

[optimizer]
lr = 1e-4  # Higher LR than fine-tuning
weight_decay = 0.1

[lr_scheduler]
scheduler_type = "universal"
warm_steps = 2000
main_decay_type = "cosine"
main_decay_ratio = 0.1
cooldown_steps = 1000
```

## Architecture Modifications

You can modify the Qwen3 architecture while loading compatible pretrained weights. Only compatible layers will be loaded; new or modified layers will be randomly initialized.

### Example: Adding Custom Layers

1. **Modify `model/model.py`**:
   - Add your custom layers to the `Qwen3` class
   - Update the forward pass

2. **Update `model/state_dict_adapter.py`**:
   - Add mappings for new layers (if loading from HF)
   - Map incompatible layers to `None` to skip loading

3. **Load pretrained weights**:
   - Compatible layers (attention, FFN, embeddings) will load from HF
   - New/modified layers will initialize randomly

4. **Train**:
   ```bash
   python train.py --job.config_file=configs/qwen3_custom.toml
   ```

## Titan-Sci Integration

The qwen3_custom model is fully integrated with titan-sci infrastructure:

### Dataloader: sci_dataloader

Automatically configured via the `[data]` section in your config:

```toml
[data]
dataloader = "MMapDataset"  # or "ChunkedMMapDataset"
min_doc_len = 10
seed = 42
# data_prefix and chunks_dir are injected by cluster_config.py
```

No model-specific code needed!

### Learning Rate Scheduler: Universal LR Scheduler

Supports multiple scheduler types via the `[lr_scheduler]` section:

```toml
[lr_scheduler]
scheduler_type = "universal"
warm_steps = 1000
main_decay_type = "cosine"
main_decay_ratio = 0.1
lr_min_absolute = 1e-5
cooldown_steps = 500
```

Available scheduler types:
- `"wsd"`: Warmup-Stable-Decay (torchtitan default)
- `"cosine"`: Cosine annealing with warmup
- `"universal"`: 3-phase scheduler (warm -> main -> cooldown) with flexible control

### Validation

Configured via the `[validation]` section:

```toml
[validation]
enable = true
freq = 1000  # Validate every 1000 steps
steps = -1  # Full validation set (-1) or limit steps (>0)
max_eval_samples = 50000  # Total samples across all workers
local_batch_size = -1  # Use training batch size
```

### Parameter Logging

Log parameter statistics to TensorBoard:

```toml
[parameter_logging]
enabled = true
log_interval = 1000
log_parameters = true  # Log param stats (max, min, norm, std)
log_gradients = true  # Log gradient stats
log_optimizer_states = true  # Log optimizer stats
```

## Training Examples

### Example 1: Fine-tune Qwen3 0.6B

```bash
# 1. Download model
python torchtitan/scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-0.6B \
    --assets model tokenizer

# 2. Create config (configs/finetune_qwen3_0.6B.toml)
cat > configs/finetune_qwen3_0.6B.toml << 'EOF'
[job]
experiment_folder = "train_qwen3_0.6B"  # Resolved under $OUTPUT_DIR

[model]
name = "qwen3_custom"
flavor = "0.6B"
vocab_size = 151936
tokenizer_path = "./assets/hf/Qwen3-0.6B"

[training]
dataset = "sci_dataset"
local_batch_size = 8
seq_len = 2048
steps = 10000
mixed_precision_param = "bfloat16"

[optimizer]
name = "AdamW"
lr = 1e-5
weight_decay = 0.1

[lr_scheduler]
scheduler_type = "cosine"
warm_steps = 500
main_decay_type = "cosine"
main_decay_ratio = 0.1
lr_min_absolute = 1e-6

[checkpoint]
enable = true
folder = "checkpoint"
interval = 2000
initial_load_in_hf = true
initial_load_model_only = true
initial_load_path = "./assets/hf/Qwen3-0.6B"

[experimental]
custom_import = "titan_oellm.models"
EOF

# 3. Start training
python train.py --job.config_file=configs/finetune_qwen3_0.6B.toml
```

### Example 2: Continual Pretraining with Larger Model

```bash
# 1. Download model
python torchtitan/scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-1.7B \
    --assets model tokenizer

# 2. Start training with modified config
python train.py \
    --job.config_file=configs/qwen3_custom.toml \
    --model.flavor=1.7B \
    --model.tokenizer_path=./assets/hf/Qwen3-1.7B \
    --checkpoint.initial_load_in_hf=true \
    --checkpoint.initial_load_path=./assets/hf/Qwen3-1.7B \
    --training.steps=100000 \
    --optimizer.lr=1e-4
```

## Troubleshooting

### Issue: "Model size mismatch"

**Solution**: Ensure `model.flavor` matches the HuggingFace model size.
- For `Qwen/Qwen3-0.6B`, use `flavor = "0.6B"`
- For `Qwen/Qwen3-1.7B`, use `flavor = "1.7B"`

### Issue: "Tokenizer vocab size mismatch"

**Solution**: Qwen3 uses `vocab_size = 151936`. Ensure this is set in your config:

```toml
[model]
vocab_size = 151936
```

### Issue: "Checkpoint not found"

**Solution**: Verify the HF checkpoint path:
1. Check the path exists: `ls ./assets/hf/Qwen3-0.6B`
2. Ensure `initial_load_path` points to the correct directory
3. The directory should contain `model.safetensors` or `pytorch_model.bin`

### Issue: "State dict keys don't match"

**Solution**: The `Qwen3StateDictAdapter` handles HF → torchtitan conversion automatically. If you see this error:
1. Check that you're using `initial_load_in_hf = true`
2. Verify the HF model is actually a Qwen3 model
3. Check for architecture modifications in your custom model

## Advanced Usage

### Custom Model Size

Define a custom model size in `__init__.py:qwen3_custom_configs`:

```python
qwen3_custom_configs = {
    # ... existing configs ...
    "custom_2B": Qwen3CustomModelArgs(
        dim=2048,
        n_layers=32,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=8192,
        # ... other params ...
    ),
}
```

Then use it in your config:

```toml
[model]
flavor = "custom_2B"
```

### Distributed Training

Use data parallelism and tensor parallelism for larger models:

```toml
[parallelism]
data_parallel_shard_degree = -1  # Use all available GPUs
tensor_parallel_degree = 2  # Shard model across 2 GPUs

[compile]
enable = true
backend = "inductor"
```

### Mixed Precision Training

```toml
[training]
mixed_precision_param = "bfloat16"  # or "float16"

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true  # FP8 for even more memory savings
```

## References

- **Qwen3 Official**: https://github.com/QwenLM/Qwen3
- **HuggingFace Models**: https://huggingface.co/collections/Qwen/qwen3-6751c5cbf6fc98b0838a3d2f
- **TorchTitan**: https://github.com/pytorch/torchtitan
- **Titan-Sci Documentation**: `../../../README.md`

## License

This implementation is based on TorchTitan's Qwen3 model, which is licensed under the BSD-style license. See `LICENSE` in the repository root.
