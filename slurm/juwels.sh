#!/bin/bash -x
#
# JUWELS SLURM Training Script
#
# Usage:
#   bash submit_job.sh slurm/juwels.sh                                    # Default: juwels + norm_gpt + slimpajama + neox
#   bash submit_job.sh slurm/juwels.sh --model.flavor=1B --training.steps=20000  # Override parameters
#
#   DATASET=fineweb_edu bash submit_job.sh slurm/juwels.sh               # Use different dataset
#   TOKENIZER=llama3 bash submit_job.sh slurm/juwels.sh                  # Use different tokenizer
#   CONFIG=base_plus.toml bash submit_job.sh slurm/juwels.sh             # Use gpt_plus model
#   CLUSTER=jupiter bash submit_job.sh slurm/juwels.sh                   # Use different cluster paths (for testing)
#   TITAN_USER=your_username bash submit_job.sh slurm/juwels.sh          # Use your user config (REQUIRED)
#
# Environment variables:
#   TITAN_USER - Username for user-specific configs (REQUIRED)
#   CLUSTER    - Cluster name from cluster_paths.toml (default: juwels)
#   DATASET    - Dataset name from cluster_paths.toml (default: slimpajama_627b)
#   TOKENIZER  - Tokenizer name from cluster_paths.toml (default: neox)
#   CONFIG     - Base config file (default: base_plus.toml)
#
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=llmApp
#SBATCH --account=transfernetx
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --time=1:00:00
#SBATCH --output=/p/scratch/.../slurm/mpi-out.%j
#SBATCH --error=/p/scratch/.../slurm/mpi-err.%j

# [Keep your NCCL exports as they are]
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=100
export NCCL_IB_RETRY_CNT=20
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET


# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo,+inductor"   # adjust to your build's accepted env vars

# Dataset and Tokenizer configuration - set via environment or use defaults
export TITAN_USER="${TITAN_USER:-joerg}"     # Username for user-specific configs (user/{joerg,korbi})
CLUSTER="juwels"            # Cluster name (juwels, jupiter, capella)
DATASET="${DATASET:-slimpajama_627b}"    # Dataset name from cluster_paths.toml
TOKENIZER="${TOKENIZER:-neox}"           # Tokenizer name from cluster_paths.toml
CONFIG="${CONFIG:-base_norm.toml}"       # Base config file (base_norm.toml or base_plus.toml)
CONTAINER="${CONTAINER:-titan_juwels_0.2.1.sif}"

# Project and container configuration
PROJECT_DIR=$(pwd)                       # Assume script is run from project root



# ============================================================================
# LOAD CLUSTER CONFIGURATION
# ============================================================================
# Load cluster-specific cache directories and paths from user-specific
# cluster_paths.toml via cluster_config.py
#
# This runs inside the container where torchtitan is available
echo "Loading cluster configuration for user '$TITAN_USER' on cluster '$CLUSTER'..."

# Load all cluster configuration as environment variables (run in container)
eval "$(apptainer exec \
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

# Additional HuggingFace settings (not in cluster config)
export HF_HUB_OFFLINE="1"
export HF_ALLOW_CODE_EVAL="1"

echo "Configuration loaded successfully:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CONTAINER: $CONTAINER"
echo "  Cache base: $(dirname $TRITON_CACHE_DIR)"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "  TORCH_INDUCTOR_CUDAGRAPH_DISABLE: $TORCH_INDUCTOR_CUDAGRAPH_DISABLE"

# Create cache directories if they don't exist
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"

# CUDA allocator: reduce fragmentation and large-alloc failures
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Disable cudagraphs unless explicitly enabled (can increase memory usage)
: "${TORCH_INDUCTOR_CUDAGRAPH_DISABLE:=1}"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# Resolve IPv4 addresses explicitly (JUWELS has dual-stack; torchrun needs IPv4 here)
MASTER_IP=$(getent hosts ${nodes[0]} | awk '{print $1}' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -n1)
if [ -z "$MASTER_IP" ]; then
    echo "ERROR: Failed to resolve IPv4 for master node ${nodes[0]}"
    exit 1
fi

echo "Master IP: $MASTER_IP"
echo "Nodes: ${nodes[@]}"

ml Stages/2026
ml GCC/14.3.0
ml Python/3.13.5
ml CUDA/13
ml NCCL/default-CUDA-13

APPTAINER="apptainer exec --nv \
    --pwd /opt/titan-oellm \
    --env TITAN_USER=$TITAN_USER \
    --env NCCL_SOCKET_IFNAME=ib0 \
    --env GLOO_SOCKET_IFNAME=ib0 \
    --env MASTER_ADDR=$MASTER_IP \
    --env MASTER_PORT=29500 \
    --env TORCH_HOME=$TORCH_HOME \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env TORCH_INDUCTOR_CUDAGRAPH_DISABLE=$TORCH_INDUCTOR_CUDAGRAPH_DISABLE \
    --env PYTHONPATH=/opt/titan-oellm/torchtitan \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    --bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
    --bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
    --bind $TORCH_HOME:$TORCH_HOME \
    $PROJECT_DIR/$CONTAINER"

# Generate cluster-specific path arguments (with automatic validation)
# Note: Runs in container on master node before distributing to compute nodes
echo "Validating configuration and paths..."
CLUSTER_ARGS=$(apptainer exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-oellm \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_cli_args
print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', '$CONFIG'))
")

# Check if validation failed
if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Aborting."
    exit 1
fi
echo "Validation passed."

for (( i=0; i<$SLURM_NNODES; i++ )); do
    node=${nodes[$i]}
    node_ip=$(getent hosts ${node} | awk '{print $1}' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -n1)
    if [ -z "$node_ip" ]; then
        echo "ERROR: Failed to resolve IPv4 for node ${node}"
        exit 1
    fi

    SRUN_ARGS="--exclusive -N1 -n1 -w ${node}"

    LAUNCHER="torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --node_rank=$i \
        --master_addr=$MASTER_IP \
        --master_port=29500 \
        --local_addr=$node_ip \
        --rdzv_backend=static \
        --rdzv_endpoint=$MASTER_IP:29500"

    srun $SRUN_ARGS $APPTAINER bash -c "
        cd /opt/titan-oellm
        exec $LAUNCHER -m torchtitan.train $CLUSTER_ARGS \"\$@\"" &

    if [ $i -eq 0 ]; then
        sleep 10
    else
        sleep 2
    fi
done
wait

