#!/bin/bash -x
#
# Jupiter SLURM Training Script
#
# Environment variables:
#   TITAN_USER - Username for user-specific configs (default: user)
#   CLUSTER    - Cluster name from cluster_paths.toml (default: jupiter)
#   DATASET    - Dataset name from cluster_paths.toml (default: slimpajama_627b)
#   TOKENIZER  - Tokenizer name from cluster_paths.toml (default: neox)
#   CONFIG     - Base config file (default: base_norm.toml)
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=llmApp
#SBATCH --account=jureap59
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm/mpi-out.%j
#SBATCH --error=slurm/mpi-err.%j


export NCCL_SOCKET_TIMEOUT=60000
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=100
export NCCL_IB_RETRY_CNT=20
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

export HF_ALLOW_CODE_EVAL="1"


export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

export MASTER_PORT=20073
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Prefer the interconnect IPv4 address for multi-node rendezvous
head_node_interconnect="${head_node}-interconnect-1.jupiter.internal"
if getent ahostsv4 "$head_node_interconnect" >/dev/null 2>&1; then
    head_node_ip=$(getent ahostsv4 "$head_node_interconnect" | awk 'NR==1{print $1}')
    export MASTER_ADDR="$head_node_interconnect"
else
    head_node_ip=$(getent ahostsv4 "$head_node" | awk 'NR==1{print $1}')
    export MASTER_ADDR="$head_node_ip"
fi
echo "INFO: Head Node IP: $head_node_ip"
echo "INFO: Head Node: $head_node"

# Single-node: use localhost — the management IP (10.128.x.x) is not bindable
if [ "$SLURM_NNODES" -eq 1 ]; then
    export MASTER_ADDR="127.0.0.1"
fi

echo "INFO: MASTER_ADDR: $MASTER_ADDR"
echo "INFO: number of nodes: $SLURM_NNODES"
echo "INFO: number of gpus per node: $GPUS_PER_NODE"
echo "INFO: number of SLURM_JOB_ID: $SLURM_JOB_ID"
echo "INFO: number of SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "INFO: SLURM_PROCID  $SLURM_PROCID"

NUM_GPUS_PER_NODE=4
SLURM_CPUS_PER_TASK=18

# Dataset and Tokenizer configuration - set via environment or use defaults
export TITAN_USER="${TITAN_USER:-user}"     # Username for user-specific configs
CLUSTER="jupiter"                        # Cluster name
DATASET="${DATASET:-slimpajama_627b}"    # Dataset name from cluster_paths.toml
TOKENIZER="${TOKENIZER:-neox}"           # Tokenizer name from cluster_paths.toml
CONFIG="${CONFIG:-base_plus.toml}"       # Base config file
CONTAINER="${CONTAINER:-titan_jupiter_0.2.1.sif}"     

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
    --bind $PROJECT_DIR:/opt/titan-sci \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-sci')
from titan_sci.cluster_config import get_env_exports
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

# CUDA allocator: reduce fragmentation and large-alloc failures
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Disable cudagraphs unless explicitly enabled (can increase memory usage)
: "${TORCH_INDUCTOR_CUDAGRAPH_DISABLE:=1}"

ml Stages/2026
ml GCC/14.3.0
ml Python/3.13.5
ml CUDA/13
ml NCCL/default-CUDA-13


SRUN_ARGS="
    --nodes=$SLURM_NNODES \
    --gres=gpu:$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --wait=20 \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"


APPTAINER="apptainer exec \
--writable-tmpfs \
--env TITAN_USER=$TITAN_USER \
--env MASTER_ADDR=$MASTER_ADDR \
--env MASTER_PORT=$MASTER_PORT \
--env NCCL_SOCKET_IFNAME=ib0 \
--env GLOO_SOCKET_IFNAME=ib0 \
--env NCCL_SOCKET_FAMILY=AF_INET \
--env GLOO_SOCKET_FAMILY=AF_INET \
--env TORCH_HOME=$TORCH_HOME \
--env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
--env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
--env TORCHINDUCTOR_FX_GRAPH_CACHE=$TORCHINDUCTOR_FX_GRAPH_CACHE \
--env TORCHINDUCTOR_AUTOGRAD_CACHE=$TORCHINDUCTOR_AUTOGRAD_CACHE \
--env TORCH_INDUCTOR_CUDAGRAPH_DISABLE=$TORCH_INDUCTOR_CUDAGRAPH_DISABLE \
--env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
--env LD_LIBRARY_PATH=/opt/nvidia/lib:/usr/local/cuda/lib64 \
--env LIBRARY_PATH=/opt/nvidia/lib:/usr/local/cuda/lib64 \
--env TRITON_LIBCUDA_PATH=/opt/nvidia/lib/libcuda.so.1 \
--env PYTHONPATH=/opt/titan-sci/torchtitan \
--bind $PROJECT_DIR:/opt/titan-sci \
--bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
--bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
--bind $TORCH_HOME:$TORCH_HOME \
--bind /usr/lib64/libcuda.so.1:/opt/nvidia/lib/libcuda.so.1:ro \
--bind /usr/lib64/libcuda.so:/opt/nvidia/lib/libcuda.so:ro \
--bind /usr/lib64/libnvidia-ml.so.1:/opt/nvidia/lib/libnvidia-ml.so.1:ro \
--bind /usr/lib64/libnvidia-ptxjitcompiler.so.1:/opt/nvidia/lib/libnvidia-ptxjitcompiler.so.1:ro \
--nv $PROJECT_DIR/$CONTAINER "

# Generate cluster-specific path arguments (with automatic validation)
# Note: Runs in container on master node before distributing to compute nodes
echo "Validating configuration and paths..."
CLUSTER_ARGS=$(apptainer exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-sci \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-sci')
from titan_sci.cluster_config import get_cli_args
print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', '$CONFIG', '/opt/titan-sci/titan_sci/configs'))
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
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT"

srun $SRUN_ARGS $APPTAINER bash -c "
    echo SLURM_LOCALID \$SLURM_LOCALID
    echo SLURM_PROCID \$SLURM_PROCID \$SLURM_NTASKS
    exec $LAUNCHER -m torchtitan.train --job.config_file=/opt/titan-sci/titan_sci/configs/$CONFIG $CLUSTER_ARGS $*
"
