#!/bin/bash
# SLURM Job Submission Wrapper / Local Training Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# FUNCTIONS
# ============================================================================

die() { echo "Error: $*" >&2; exit 1; }

show_help() {
    cat << 'EOF'
Usage:
  bash submit_job.sh [sbatch-options] -- [training-args...]
  bash submit_job.sh [sbatch-options] [slurm-script] [training-args...]
  bash submit_job.sh --local [training-args...]

Options:
  --help, -h   Show this help
  --dry-run    Print command without executing
  --local      Run locally without SLURM
  --           Separator between sbatch options and training arguments

Environment:
  TITAN_USER   Username for configs (REQUIRED)
  CLUSTER      Cluster name (default: auto-detect from hostname or 'local')
  DATASET      Dataset name (default: test_dataset)
  TOKENIZER    Tokenizer name (default: neox)
  CONFIG       Config file path for --local mode
  NPROC        Number of GPUs for local (default: 1)

Examples:
  # Submit with auto-detected cluster (hostname-based)
  TITAN_USER=joerg bash submit_job.sh --nodes=1 --time=1:00:00 -- --model.flavor=1.8B

  # Submit with explicit cluster
  TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh --nodes=4 -- --compile.mode=max-autotune

  # Local training
  TITAN_USER=joerg bash submit_job.sh --local --model.flavor=debugmodel
EOF
}

find_container() {
    local cluster=$1
    local container="$SCRIPT_DIR/titan_${cluster}_0.2.1.sif"
    [[ -f "$container" ]] || die "Container not found: $container"
    echo "$container"
}

detect_container_runtime() {
    if command -v apptainer &> /dev/null; then
        echo "apptainer"
    elif command -v singularity &> /dev/null; then
        echo "singularity"
    else
        die "Neither apptainer nor singularity found in PATH"
    fi
}

test_container_runtime() {
    local runtime=$(detect_container_runtime)
    local container=$1

    # Test if container runtime works (some clusters have permission issues on login nodes)
    if ! $runtime exec "$container" /bin/true 2>/dev/null; then
        return 1
    fi
    return 0
}

run_python() {
    local container=$1 code=$2
    local runtime=$(detect_container_runtime)
    local cluster="${CLUSTER:-}"

    # PyTorch's DynamoCache initializes at import time, so we need to set
    # cache env vars to a writable location before any torch imports
    local cache_env=(
        --env TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
        --env TORCH_HOME=/tmp/torch_home
    )

    # Capella's Singularity on login nodes has issues with bind mounts
    # Use --pwd and rely on automatic home directory binding instead
    if [[ "$cluster" == "capella" && "$runtime" == "singularity" ]]; then
        (cd "$SCRIPT_DIR" && $runtime exec \
            --pwd "$SCRIPT_DIR" \
            --env TITAN_USER="$TITAN_USER" \
            "${cache_env[@]}" \
            "$container" python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
$code")
    else
        # Use bind mounts for other clusters
        $runtime exec \
            --env TITAN_USER="$TITAN_USER" \
            "${cache_env[@]}" \
            --bind "$SCRIPT_DIR:/opt/titan-oellm" \
            "$container" python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
$code"
    fi
}

detect_cluster_from_script() {
    case "$1" in
        *juwels*)  echo "juwels" ;;
        *capella*) echo "capella" ;;
        *jupiter*) echo "jupiter" ;;
        *) die "Cannot detect cluster from: $1. Set CLUSTER explicitly." ;;
    esac
}

detect_cluster_from_hostname() {
    local hostname=$(hostname)
    case "$hostname" in
        jpbl*)     echo "jupiter" ;;
        jwlogin*)  echo "juwels" ;;
        c[0-9]*)   echo "capella" ;;  # Capella nodes: c1, c2, etc.
        *capella*) echo "capella" ;;
        *) echo "" ;;  # Return empty if cannot detect
    esac
}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

DRY_RUN=false
LOCAL_MODE=false
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h) show_help; exit 0 ;;
        --dry-run) DRY_RUN=true ;;
        --local)   LOCAL_MODE=true ;;
        *)         ARGS+=("$arg") ;;
    esac
done

[[ -n "${TITAN_USER:-}" ]] || die "TITAN_USER not set. Run: export TITAN_USER=your_username"

# ============================================================================
# LOCAL EXECUTION MODE
# ============================================================================

if [[ "$LOCAL_MODE" == true ]]; then
    # Auto-detect cluster from hostname if not set
    if [[ -z "${CLUSTER:-}" ]]; then
        CLUSTER=$(detect_cluster_from_hostname)
        if [[ -n "$CLUSTER" ]]; then
            echo "Auto-detected cluster '$CLUSTER' from hostname $(hostname)"
        else
            CLUSTER="local"
        fi
    fi
    DATASET="${DATASET:-test_dataset}"
    TOKENIZER="${TOKENIZER:-neox}"
    NPROC="${NPROC:-1}"
    CONFIG="${CONFIG:-user/$TITAN_USER/configs/debug.toml}"
    CONTAINER=$(find_container "$CLUSTER")
    
    # Test if container runtime works (fails on some login nodes like Capella)
    if ! test_container_runtime "$CONTAINER"; then
        cat >&2 << EOF
ERROR: Container runtime not available on this node (permission issues).

On Capella, Singularity only works on compute nodes, not login nodes.
Please use one of these alternatives:

1. Submit as a SLURM job (recommended):
   TITAN_USER=$TITAN_USER DATASET=$DATASET TOKENIZER=$TOKENIZER CONFIG=$CONFIG \\
   bash submit_job.sh --nodes=1 --time=0:30:00 -- ${ARGS[@]}

2. Get an interactive compute node first:
   salloc --nodes=1 --gres=gpu:1 --time=0:30:00
   # Then run your --local command on the compute node

3. Use the direct slurm script:
   sbatch --nodes=1 --time=0:30:00 slurm/capella.sh ${ARGS[@]}
EOF
        exit 1
    fi

    # Format config file path for container
    # If CONFIG is just a filename (no /), prepend titan_oellm/configs/
    if [[ "$CONFIG" != */* ]]; then
        CONFIG_ARG="--job.config_file=/opt/titan-oellm/titan_oellm/configs/$CONFIG"
    else
        CONFIG_ARG="--job.config_file=/opt/titan-oellm/$CONFIG"
    fi
    
    echo "=== Local Training: cluster=$CLUSTER dataset=$DATASET tokenizer=$TOKENIZER gpus=$NPROC ==="
    
    # Load environment from cluster config
    eval "$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_env_exports; print(get_env_exports('$CLUSTER'))")"
    [[ -n "${TRITON_CACHE_DIR:-}" ]] || die "Failed to load cluster config for '$CLUSTER'"

    # Set OUTPUT_DIR from cluster config
    OUTPUT_DIR=$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_cluster_config; config = get_cluster_config('$CLUSTER', '$TITAN_USER'); print(config['output_dir'])")

    # Get dataset/tokenizer CLI args (skip validation since paths may not be accessible from login node)
    CLUSTER_ARGS=$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_cli_args; print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', validate=False))")
    
    # Create directories
    mkdir -p "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$OUTPUT_DIR"

    # Load CUDA modules for JUWELS (required for --nv flag to work properly)
    if [[ "$CLUSTER" == "juwels" ]]; then
        echo "Loading JUWELS CUDA modules..."
        ml Stages/2026
        ml GCC/14.3.0
        ml Python/3.13.5
        ml CUDA/13
        ml NCCL/default-CUDA-13
    fi

    if [[ "$CLUSTER" == "jupiter" ]]; then
        echo "Loading JUWELS CUDA modules..."
        ml Stages/2026
        ml GCC/14.3.0
        ml Python/3.13.5
        ml CUDA/13
        ml NCCL/default-CUDA-13
    fi





    # Build command
    CONTAINER_RUNTIME=$(detect_container_runtime)

    # Local cluster uses --nvccli for better CUDA compatibility
    if [[ "$CLUSTER" == "local" ]]; then
        NV_FLAG="--nvccli"
    else
        NV_FLAG="--nv"
    fi

    CMD=(
        $CONTAINER_RUNTIME exec $NV_FLAG
        --bind "$TRITON_CACHE_DIR:$TRITON_CACHE_DIR"
        --bind "$SCRIPT_DIR:/opt/titan-oellm"
        --bind "$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"
        --bind "$TORCH_HOME:$TORCH_HOME"
        --bind "$HOME:$HOME"
        --env TITAN_USER="$TITAN_USER"
        --env OUTPUT_DIR="$OUTPUT_DIR"
        --env TORCH_HOME="$TORCH_HOME"
        --env PYTHONPATH="/opt/titan-oellm/torchtitan"
        --pwd /opt/titan-oellm
    )

    # Bind DATA_DIR if set (needed for capella and other clusters with separate data paths)
    if [[ -n "${DATA_DIR:-}" ]]; then
        CMD+=(--bind "$DATA_DIR:$DATA_DIR")
    fi

    CMD+=(
        "$CONTAINER"
        torchrun --nproc_per_node="$NPROC" --nnodes=1 --node_rank=0
                 --master_addr=localhost --master_port=29500
        -m torchtitan.train "$CONFIG_ARG" $CLUSTER_ARGS "${ARGS[@]}"
    )
    
    echo "Command: ${CMD[*]}"
    [[ "$DRY_RUN" == true ]] && { echo "[DRY-RUN]"; exit 0; }
    exec "${CMD[@]}"
fi

# ============================================================================
# SLURM SUBMISSION MODE
# ============================================================================

# Setup venv for cluster config operations (avoids container dependency)
VENV_DIR="${SCRIPT_DIR}/.venv_submit"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment for cluster config..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install -r "$SCRIPT_DIR/requirements_login_node.txt"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo "Virtual environment setup complete"
else
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate virtual environment"
        exit 1
    fi
fi

# Separate SLURM options from training arguments using "--" separator
# Format: submit_job.sh [sbatch-opts] -- [training-args]
SLURM_OPTS=()
TRAINING_ARGS=()
SLURM_SCRIPT=""
FOUND_SEPARATOR=false

for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--" ]]; then
        FOUND_SEPARATOR=true
    elif [[ "$FOUND_SEPARATOR" == true ]]; then
        # Everything after "--" goes to training args
        TRAINING_ARGS+=("$arg")
    elif [[ "$arg" == *.sh ]]; then
        SLURM_SCRIPT="$arg"
    else
        # Everything before "--" (or before script) goes to SLURM options
        SLURM_OPTS+=("$arg")
    fi
done

if [[ -z "$SLURM_SCRIPT" ]]; then
    # Try to detect cluster from hostname if CLUSTER not set
    if [[ -z "${CLUSTER:-}" ]]; then
        CLUSTER=$(detect_cluster_from_hostname)
        [[ -n "$CLUSTER" ]] && echo "Auto-detected cluster '$CLUSTER' from hostname $(hostname)"
    fi

    [[ -n "${CLUSTER:-}" ]] || die "No SLURM script provided and CLUSTER not set. Set CLUSTER or provide a slurm script."
    SLURM_SCRIPT="slurm/${CLUSTER}.sh"
    [[ -f "$SCRIPT_DIR/$SLURM_SCRIPT" ]] || die "Auto-selected script not found: $SLURM_SCRIPT"
fi

# Detect cluster from script name if not set
if [[ -z "${CLUSTER:-}" ]]; then
    CLUSTER=$(detect_cluster_from_script "$SLURM_SCRIPT")
fi

CONTAINER=$(find_container "$CLUSTER")

# Get output directory from config (using venv, no container needed)
OUTPUT_DIR=$(python3 -c "
import importlib.util
import sys

# Load cluster_config directly to avoid titan_oellm.__init__.py (which imports torch)
spec = importlib.util.spec_from_file_location('cluster_config', '$SCRIPT_DIR/titan_oellm/cluster_config.py')
cluster_config = importlib.util.module_from_spec(spec)
sys.modules['cluster_config'] = cluster_config
spec.loader.exec_module(cluster_config)

config = cluster_config.get_cluster_config('$CLUSTER', '$TITAN_USER')
print(config['output_dir'])
")
[[ -n "$OUTPUT_DIR" ]] || die "Failed to get output_dir for cluster '$CLUSTER'"

export OUTPUT_DIR
echo "Cluster: $CLUSTER | Output: $OUTPUT_DIR | Container: $(basename "$CONTAINER")"

SLURM_LOG_DIR="$OUTPUT_DIR/slurm"
[[ "$DRY_RUN" == false ]] && mkdir -p "$SLURM_LOG_DIR"

# Build sbatch command: sbatch [slurm-opts] script [training-args]
CMD=(sbatch --output="$SLURM_LOG_DIR/mpi-out.%j" --error="$SLURM_LOG_DIR/mpi-err.%j" "${SLURM_OPTS[@]}" "$SLURM_SCRIPT" "${TRAINING_ARGS[@]}")
echo "Command: ${CMD[*]}"
[[ "$DRY_RUN" == true ]] && { echo "[DRY-RUN]"; exit 0; }
exec "${CMD[@]}"
