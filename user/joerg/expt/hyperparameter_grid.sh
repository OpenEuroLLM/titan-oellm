#!/bin/bash
# Hyperparameter Grid: lr × nodes × budget × betas
set -euo pipefail

# Config
export TITAN_USER=joerg
CLUSTER=capella
DATASET=nemotron_cc
TOKENIZER=nemotron
CONFIG=qwen3_custom.toml

# Fixed training params
LBS=32
GPUS=4
SEQ=4096
MODEL=125M
TPS_PER_GPU=150000

# Grid
LRS=(0.001 0.002 0.0046)
NODES=(1 2 4)
BUDGETS_B=(5 10 20 40)  # in billions
BETAS=("0.9,0.95" "0.9,0.99" "0.95,0.99")

for lr in "${LRS[@]}"; do
    for n in "${NODES[@]}"; do
        for bb in "${BUDGETS_B[@]}"; do
            for bp in "${BETAS[@]}"; do
                b1=${bp%,*}; b2=${bp#*,}
                budget=$((bb * 1000000000))
                gbs=$((LBS * GPUS * n * SEQ))
                steps=$((budget / gbs))
                name="lr${lr}_n${n}_${bb}B_b${b1}${b2}"

                # Calculate runtime: budget / (TPS_PER_GPU * total_gpus), round up to 30min + 30min buffer
                total_tps=$((TPS_PER_GPU * GPUS * n))
                runtime_min=$(( (budget / total_tps + 59) / 60 ))  # seconds to minutes, round up
                runtime_min=$(( ((runtime_min + 29) / 30) * 30 + 30 ))  # round up to 30min + 30min buffer
                TIME=$(printf "%d:%02d:00" $((runtime_min / 60)) $((runtime_min % 60)))

                ARGS="--model.flavor=$MODEL --job.dump_folder=scale_token_budget_125M/$name \
                --metrics.save_tb_folder=tb \
                --optimizer.lr=$lr --optimizer.beta1=$b1 --optimizer.beta2=$b2 \
                --training.local_batch_size=$LBS --training.seq_len=$SEQ --training.steps=$steps \
                --validation.enable=true --validation.freq=1000 --checkpoint.enable=true --checkpoint.interval=5000 \
                --parameter_logging.enabled --parameter_logging.log_interval=500 \
                --parameter_logging.log_parameters --parameter_logging.log_gradients \
                --parameter_logging.log_optimizer_states"

                TITAN_USER=$TITAN_USER DATASET=$DATASET TOKENIZER=$TOKENIZER CLUSTER=$CLUSTER CONFIG=$CONFIG \
                bash submit_job.sh --nodes=$n --time=$TIME -- $ARGS && sleep 1

done; done; done; done
