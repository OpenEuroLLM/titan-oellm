#!/bin/bash
# Experiment: gpt_plus vs qwen3_custom architecture comparison
#
# Compares training performance of two architectures at ~130M scale:
#   01. gpt_plus 130Msci    — QKNormPlus attention, SwiGLU MLP, RoPE scaling
#   02. qwen3_custom 130Msci — Qwen3-style with QK-norm, GQA, SwiGLU
#
# Both use identical optimizer, data, and training setup — only architecture differs.
# Both configs use vocab_size=50432 (neox), rope_theta=500000, weight tying.
#
# Compute: 4 nodes (16 GPUs), lbs=16, GBS=256, seq_len=4096
#          ~1.05M tokens/step, 19000 steps ≈ 20B tokens
# Dataset: nemotron_oellm with neox tokenizer

set -euo pipefail

# ============================================================================
# SETTINGS
# ============================================================================
ACCOUNT=reformo
PARTITION=booster
NODES=4
SEQ_LEN=4096
LOCAL_BATCH_SIZE=16
STEPS=19000
LR=0.004

BASE_OUTPUT="./outputs/experiments/model_comparison"
TIME="12:00:00"

GBS=$((NODES * 4 * LOCAL_BATCH_SIZE))

# ============================================================================
# Shared args
# ============================================================================
COMMON_ARGS=(
    --model.flavor=130Msci
    --model.vocab_size=50432
    --optimizer.name=AdamW
    --optimizer.lr=$LR
    --optimizer.weight_decay=0.1
    --compile.enable
    --compile.backend=inductor
    --compile.components=model,loss
    --metrics.save_tb_folder=tb
    --parameter_logging.log_parameters
    --parameter_logging.log_gradients
    --parameter_logging.no_log_optimizer_states
    --data.dataloader=DeterministicPackedDataset
    --training.local_batch_size=$LOCAL_BATCH_SIZE
    --training.seq_len=$SEQ_LEN
    --training.steps=$STEPS
    --validation.enable
    --validation.freq=1000
    --checkpoint.no_enable
    --metrics.log_freq=50
    --lr_scheduler.warm_ratio=0.1
    --lr_scheduler.warm_direction=up
    --lr_scheduler.warm_type=linear
    --lr_scheduler.main_decay_type=cosine
    --lr_scheduler.main_decay_ratio=0.1
    --lr_scheduler.cooldown_ratio=0.0
    --lr_scheduler.lr_min_absolute=1e-5
)

# ============================================================================
# MAIN
# ============================================================================
echo "Architecture Comparison: gpt_plus vs qwen3_custom (130Msci)"
echo "============================================================"
echo "Nodes: $NODES | LBS: $LOCAL_BATCH_SIZE | SeqLen: $SEQ_LEN | GBS: $GBS"
echo "Steps: $STEPS | LR: $LR"
echo "Tokens/step: ~$((GBS * SEQ_LEN / 1000000))M | Total: ~$((GBS * SEQ_LEN * STEPS / 1000000000))B"
echo ""

# --- 01. gpt_plus 130Msci ---
echo "Submitting: 01_gpt_plus_130Msci"
TITAN_USER=joerg \
DATASET=nemotron_oellm \
TOKENIZER=neox \
CONFIG=gpt_plus.toml \
bash submit_job.sh \
    --nodes=$NODES \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --time=$TIME \
    -- \
    "${COMMON_ARGS[@]}" \
    --model.tie_embedding \
    --job.dump_folder=${BASE_OUTPUT}/01_gpt_plus_130Msci/n${NODES}_lr${LR}


# --- 02. qwen3_custom 130Msci ---
echo "Submitting: 02_qwen3_custom_130Msci"
TITAN_USER=joerg \
DATASET=nemotron_oellm \
TOKENIZER=neox \
CONFIG=qwen3_custom.toml \
bash submit_job.sh \
    --nodes=$NODES \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --time=$TIME \
    -- \
    "${COMMON_ARGS[@]}" \
    --model.enable_weight_tying=True \
    --job.dump_folder=${BASE_OUTPUT}/02_qwen3_custom_130Msci/n${NODES}_lr${LR}


