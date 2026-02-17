#!/bin/bash -x
#
# Capella SLURM Training Script
#
# Usage:
#   bash submit_job.sh slurm/capella.sh                                    # Default: capella + norm_gpt + slimpajama + neox
#   bash submit_job.sh slurm/capella.sh --model.flavor=1B --training.steps=20000  # Override parameters
#
#   DATASET=fineweb_edu bash submit_job.sh slurm/capella.sh               # Use different dataset
#   TOKENIZER=llama3 bash submit_job.sh slurm/capella.sh                  # Use different tokenizer
#   CONFIG=base_plus.toml bash submit_job.sh slurm/capella.sh             # Use gpt_plus model
#   TITAN_USER=korbi bash submit_job.sh slurm/capella.sh                  # Use different user's config (default: joerg)
#
# Environment variables:
#   TITAN_USER - Username for user-specific configs (default: joerg)
#   CLUSTER    - Cluster name from cluster_paths.toml (default: capella)
#   DATASET    - Dataset name from cluster_paths.toml (default: slimpajama_627b)
#   TOKENIZER  - Tokenizer name from cluster_paths.toml (default: neox)
#   CONFIG     - Base config file (default: base_norm.toml)
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=54
#SBATCH --job-name=llmApp
#SBATCH --threads-per-core=1
#SBATCH --time=1:00:00
#SBATCH --output=/data/.../slurm/mpi-out.%j
#SBATCH --error=/data/.../slurm/mpi-err.%j

# ============================================================================
# SETUP
# ============================================================================
PROJECT_DIR=$(pwd)  # Assume script is run from project root

export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=100
export NCCL_IB_RETRY_CNT=20
export NCCL_ALGO=Ring

# Additional NCCL robustness settings for multi-node
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export HF_ALLOW_CODE_EVAL="1"

export NCCL_SOCKET_IFNAME=eno4

# Torchrun rendezvous timeout (seconds) - increase for large node counts
export TORCH_DISTRIBUTED_TIMEOUT=1800


export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Dynamic port selection to avoid conflicts from previous jobs
# Use job ID to generate a unique port in range 20000-30000
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export NUM_NODES=$SLURM_JOB_NUM_NODES

export NUM_GPUS_PER_NODE=4
export SLURM_CPUS_PER_TASK=14
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(getent hosts $MASTER_ADDR | awk '{print $1}')
echo "INFO: Head Node IP: $head_node_ip"
echo "head_node: $head_node"


export MASTER_ADDR=$head_node_ip

echo "MASTER_ADDR: $MASTER_ADDR"

echo "INFO: number of nodes: $SLURM_NNODES"
echo "INFO: number of gpus per node: $NUM_GPUS_PER_NODE"
echo "INFO: number of SLURM_JOB_ID: $SLURM_JOB_ID"
echo "INFO: number of SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "INFO: SLURM_PROCID  $SLURM_PROCID"



# Dataset and Tokenizer configuration - set via environment or use defaults
export if [ -z "$TITAN_USER" ]; then
    echo "Error: TITAN_USER environment variable not set."
    echo "Set it before running: export TITAN_USER=your_username"
    exit 1
fi
export TITAN_USER     # Username for user-specific configs (user/{joerg,korbi})
CLUSTER="${CLUSTER:-capella}"            # Cluster name (juwels, jupiter, capella)
DATASET="${DATASET:-slimpajama_627b}"    # Dataset name from cluster_paths.toml
TOKENIZER="${TOKENIZER:-neox}"           # Tokenizer name from cluster_paths.toml
CONFIG="${CONFIG:-base_plus.toml}"       # Base config file (base_norm.toml or base_plus.toml)

# Project and container configuration
PROJECT_DIR=$(pwd)                       # Assume script is run from project root
CONTAINER="titan_juwels_0.2.0.sif"       # Container filename (needs to be build on different system)

# ============================================================================
# LOAD CLUSTER CONFIGURATION
# ============================================================================
# Load cluster-specific cache directories and paths from user-specific
# cluster_paths.toml via cluster_config.py
#
# This runs inside the container where torchtitan is available
echo "Loading cluster configuration for user '$TITAN_USER' on cluster '$CLUSTER'..."

# Load all cluster configuration as environment variables (run in container)
eval "$(singularity exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_env_exports
try:
    print(get_env_exports('$CLUSTER'))
except Exception as e:
    print(f'echo \"Error loading cluster config: {e}\"', file=sys.stderr)
    sys.exit(1)
")"

# Check if configuration was loaded successfully
if [ -z "$TRITON_CACHE_DIR" ]; then
    echo "ERROR: Failed to load cluster configuration"
    echo "Please ensure user/$TITAN_USER/cluster_paths.toml has [cluster.$CLUSTER] section"
    exit 1
fi

echo "Configuration loaded successfully:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CONTAINER: $CONTAINER"
echo "  Cache base: $(dirname $TRITON_CACHE_DIR)"

# Create cache directories if they don't exist
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$DATA_DIR"

ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12


SRUN_ARGS="
    --nodes=$SLURM_NNODES \
    --gres=gpu:$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"


APPTAINER="singularity exec --nv \
    --pwd /opt/titan-oellm \
    --env TITAN_USER=$TITAN_USER \
    --env TORCH_HOME=$TORCH_HOME \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --bind $DATA_DIR:$DATA_DIR \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    --bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
    --bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
    --bind $TORCH_HOME:$TORCH_HOME \
    $PROJECT_DIR/$CONTAINER"


# Generate cluster-specific path arguments (with automatic validation)
# Note: Runs in container on master node before distributing to compute nodes
echo "Validating configuration and paths..."
CLUSTER_ARGS=$(singularity exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    --bind $DATA_DIR:$DATA_DIR \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_cli_args
print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', '$CONFIG', '/opt/titan-oellm/titan_oellm/configs'))
")

# Check if validation failed
if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Aborting."
    exit 1
fi
echo "Validation passed."

LAUNCHER="torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $NUM_GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --max_restarts 3 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf timeout=1800,read_timeout=900"


srun $SRUN_ARGS $APPTAINER bash -c '
    # Create rank-specific cache to avoid compilation race conditions
    RANK_TRITON_CACHE="${TRITON_CACHE_DIR}/rank_${SLURM_PROCID}"
    mkdir -p "${RANK_TRITON_CACHE}"
    export TRITON_CACHE_DIR="${RANK_TRITON_CACHE}"
    export TORCHINDUCTOR_CACHE_DIR="${RANK_TRITON_CACHE}"
    exec '"$LAUNCHER -m torchtitan.train --job.config_file=/opt/titan-oellm/titan_oellm/configs/$CONFIG $CLUSTER_ARGS $@"
