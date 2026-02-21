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

export HF_ALLOW_CODE_EVAL="1"


export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Dynamic port selection to avoid conflicts from previous jobs
# Use job ID to generate a unique port in range 20000-30000
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
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
export TITAN_USER="${TITAN_USER:-joerg}"     # Username for user-specific configs (user/{joerg,korbi})
CLUSTER="${CLUSTER:-capella}"            # Cluster name (juwels, jupiter, capella)
DATASET="${DATASET:-slimpajama_627b}"    # Dataset name from cluster_paths.toml
TOKENIZER="${TOKENIZER:-neox}"           # Tokenizer name from cluster_paths.toml
CONFIG="${CONFIG:-base_norm.toml}"       # Base config file (base_norm.toml or base_plus.toml)

# Container configuration (used on compute nodes)
PROJECT_DIR=$(pwd)
CONTAINER="titan_capella_0.2.1.sif"      # Container filename

# ============================================================================
# LOAD CLUSTER CONFIGURATION (using venv on login node)
# ============================================================================
# Load cluster-specific cache directories and paths from user-specific
# cluster_paths.toml via cluster_config.py
# This now runs in the venv on the login node

echo "Loading cluster configuration for user '$TITAN_USER' on cluster '$CLUSTER'..."
echo "DEBUG: Positional arguments (\$@): $@"
echo "DEBUG: Number of arguments: $#"

# Load cluster configuration using container with writable cache paths
eval "$(singularity exec \
    --env TITAN_USER=$TITAN_USER \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    --env TORCH_HOME=/tmp/torch_home \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_env_exports
try:
    print(get_env_exports('$CLUSTER'))
except Exception as e:
    import sys
    print(f'echo \"Error loading cluster config: {e}\"', file=sys.stderr)
    sys.exit(1)
")"

# Check if configuration was loaded successfully
if [ -z "$TRITON_CACHE_DIR" ]; then
    echo "ERROR: Failed to load cluster configuration"
    echo "Please ensure user/$TITAN_USER/cluster_paths.toml has [cluster.$CLUSTER] section"
    exit 1
fi

# Get OUTPUT_DIR from cluster config
OUTPUT_DIR=$(singularity exec \
    --env TITAN_USER=$TITAN_USER \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    --env TORCH_HOME=/tmp/torch_home \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_cluster_config
try:
    config = get_cluster_config('$CLUSTER')
    print(config['output_dir'])
except Exception as e:
    print(f'echo \"ERROR: Failed to get OUTPUT_DIR: {e}\"', file=sys.stderr)
    sys.exit(1)
")
export OUTPUT_DIR

# Verify OUTPUT_DIR was set correctly
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR is empty. Failed to load from cluster configuration."
    echo "Please check user/$TITAN_USER/cluster_paths.toml has [cluster.$CLUSTER] with output_dir set."
    exit 1
fi

echo "Configuration loaded successfully:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CONTAINER: $CONTAINER"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  Cache base: $(dirname $TRITON_CACHE_DIR)"

# Create cache and output directories if they don't exist
echo "Creating directories..."
mkdir -p "$TRITON_CACHE_DIR" || echo "WARNING: Failed to create TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE" || echo "WARNING: Failed to create HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME" || echo "WARNING: Failed to create TORCH_HOME: $TORCH_HOME"
mkdir -p "$DATA_DIR" || echo "WARNING: Failed to create DATA_DIR: $DATA_DIR"
mkdir -p "$OUTPUT_DIR" || echo "WARNING: Failed to create OUTPUT_DIR: $OUTPUT_DIR"

# Verify directories exist and are writable
echo "Verifying cache directories:"
ls -ld "$TRITON_CACHE_DIR" 2>&1 || echo "ERROR: TRITON_CACHE_DIR does not exist: $TRITON_CACHE_DIR"
ls -ld "$HF_DATASETS_CACHE" 2>&1 || echo "ERROR: HF_DATASETS_CACHE does not exist: $HF_DATASETS_CACHE"
ls -ld "$TORCH_HOME" 2>&1 || echo "ERROR: TORCH_HOME does not exist: $TORCH_HOME"
ls -ld "$DATA_DIR" 2>&1 || echo "ERROR: DATA_DIR does not exist: $DATA_DIR"

# Test write access
touch "$TRITON_CACHE_DIR/.test_write" 2>&1 && rm "$TRITON_CACHE_DIR/.test_write" || echo "ERROR: TRITON_CACHE_DIR not writable: $TRITON_CACHE_DIR"

ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12


SRUN_ARGS="
    --nodes=$SLURM_NNODES \
    --gres=gpu:$NUM_GPUS_PER_NODE \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"



# Debug: Show cache environment variables
echo "Cache environment variables:"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

APPTAINER="singularity exec --nv \
--writable-tmpfs \
    --pwd /opt/titan-oellm \
    --env TITAN_USER=$TITAN_USER \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env NCCL_ALGO=$NCCL_ALGO \
    --env NCCL_TIMEOUT=$NCCL_TIMEOUT \
    --env NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT \
    --env NCCL_IB_RETRY_CNT=$NCCL_IB_RETRY_CNT \
    --env NCCL_SOCKET_IFNAME=ibp \
    --env GLOO_SOCKET_IFNAME=ibp \
    --env TORCH_HOME=$TORCH_HOME \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    --env TORCHINDUCTOR_FX_GRAPH_CACHE=$TORCHINDUCTOR_FX_GRAPH_CACHE \
    --env TORCHINDUCTOR_AUTOGRAD_CACHE=$TORCHINDUCTOR_AUTOGRAD_CACHE \
    --env TORCH_INDUCTOR_CUDAGRAPH_DISABLE=$TORCH_INDUCTOR_CUDAGRAPH_DISABLE \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env PYTHONPATH=/opt/titan-oellm/torchtitan \
    --env OUTPUT_DIR=$OUTPUT_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --bind $OUTPUT_DIR:$OUTPUT_DIR \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    --bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
    --bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
    --bind $TORCH_HOME:$TORCH_HOME \
    $PROJECT_DIR/$CONTAINER"



# Generate cluster-specific path arguments (validation skipped - paths checked at runtime)
# Note: Data paths may not be accessible yet, so validation is deferred
echo "Generating cluster arguments..."

CLUSTER_ARGS=$(singularity exec \
    --env TITAN_USER=$TITAN_USER \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    --env TORCH_HOME=/tmp/torch_home \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    --bind $DATA_DIR:$DATA_DIR \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_cli_args
print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', '$CONFIG', validate=False))
")

# Check if command succeeded
if [ $? -ne 0 ]; then
    echo "Failed to generate cluster arguments. Aborting."
    exit 1
fi
echo "Cluster arguments generated successfully."

LAUNCHER="torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $NUM_GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --max_restarts 3 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT"

echo "DEBUG: CLUSTER_ARGS=$CLUSTER_ARGS"
echo "DEBUG: Additional args (\$@)=$@"
echo "DEBUG: Full command: $LAUNCHER -m torchtitan.train $CLUSTER_ARGS $@"

srun $SRUN_ARGS $APPTAINER bash -c '
    exec '"$LAUNCHER -m torchtitan.train $CLUSTER_ARGS $@"
