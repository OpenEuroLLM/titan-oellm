#!/bin/bash
# Hyperparameter Grid: lr × nodes × budget × betas
set -euo pipefail

# Config
export TITAN_USER=joerg
CLUSTER=capella
DATASET=slimpajama_627b
TOKENIZER=neox
CONFIG=qwen3_custom.toml
TIME=0:15:00

# Fixed training params
LBS=16  # Reduced from 32 to fit in memory
GRAD_ACCUM=2  # Gradient accumulation to maintain effective batch size
GPUS=4
SEQ=4096
MODEL=125M

# Grid
LRS=(0.001)
NODES=(2)  # Start with 1 node to test, then try 2
BUDGETS_B=(5)  # in billions
BETAS=("0.9,0.95")

for lr in "${LRS[@]}"; do
    for n in "${NODES[@]}"; do
        for bb in "${BUDGETS_B[@]}"; do
            for bp in "${BETAS[@]}"; do
                b1=${bp%,*}; b2=${bp#*,}
                budget=$((bb * 1000000000))
                gbs=$((LBS * GRAD_ACCUM * GPUS * n * SEQ))
                steps=$((budget / gbs))
                name="lr${lr}_n${n}_${bb}B_b${b1}${b2}"

                # Format arguments like the working juwels script
                TITAN_USER=$TITAN_USER \
                DATASET=$DATASET \
                TOKENIZER=$TOKENIZER \
                CLUSTER=$CLUSTER \
                CONFIG=$CONFIG \
                bash submit_job.sh \
                --nodes=$n \
                --time=$TIME \
                -- \
                --model.flavor=$MODEL \
                --job.dump_folder=./outputs/scale_token_budget_125M/${name}/n${n}_lr_${lr} \
                --metrics.save_tb_folder=tb \
                --optimizer.lr=$lr \
                --optimizer.beta1=$b1 \
                --optimizer.beta2=$b2 \
                --training.local_batch_size=$LBS \
                --training.gradient_accumulation_steps=$GRAD_ACCUM \
                --training.seq_len=$SEQ \
                --training.steps=$steps \
                --validation.enable \
                --validation.freq=1000 \
                --checkpoint.enable \
                --checkpoint.interval=5000 \
                --parameter_logging.enabled \
                --parameter_logging.log_interval=500 \
                --parameter_logging.log-parameters \
                --parameter_logging.log-gradients \
                --parameter_logging.log-optimizer-states \
                && sleep 1

done; done; done; done
