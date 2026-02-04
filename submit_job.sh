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

run_python() {
    local container=$1 code=$2
    apptainer exec --env TITAN_USER="$TITAN_USER" --bind "$SCRIPT_DIR":/opt/titan-sci \
        "$container" python3 -c "
import sys; sys.path.insert(0, '/opt/titan-sci')
$code"
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

    # Format config file path for container
    # If CONFIG is just a filename (no /), prepend titan_sci/configs/
    if [[ "$CONFIG" != */* ]]; then
        CONFIG_ARG="--job.config_file=/opt/titan-sci/titan_sci/configs/$CONFIG"
    else
        CONFIG_ARG="--job.config_file=/opt/titan-sci/$CONFIG"
    fi
    
    echo "=== Local Training: cluster=$CLUSTER dataset=$DATASET tokenizer=$TOKENIZER gpus=$NPROC ==="
    
    # Load environment from cluster config
    eval "$(run_python "$CONTAINER" "from titan_sci.cluster_config import get_env_exports; print(get_env_exports('$CLUSTER'))")"
    [[ -n "${TRITON_CACHE_DIR:-}" ]] || die "Failed to load cluster config for '$CLUSTER'"

    # Set OUTPUT_DIR from cluster config
    OUTPUT_DIR=$(run_python "$CONTAINER" "from titan_sci.cluster_config import get_cluster_config; config = get_cluster_config('$CLUSTER', '$TITAN_USER'); print(config['output_dir'])")
    
    # Get dataset/tokenizer CLI args
    CLUSTER_ARGS=$(run_python "$CONTAINER" "from titan_sci.cluster_config import get_cli_args; print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER'))")
    
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
    CMD=(
        apptainer exec --nv
        --bind "$TRITON_CACHE_DIR:$TRITON_CACHE_DIR"
        --bind "$SCRIPT_DIR:/opt/titan-sci"
        --bind "$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"
        --bind "$TORCH_HOME:$TORCH_HOME"
        --bind "$HOME:$HOME"
        --env TITAN_USER="$TITAN_USER"
        --env OUTPUT_DIR="$OUTPUT_DIR"
        --env TORCH_HOME="$TORCH_HOME"
        --env PYTHONPATH="/opt/titan-sci/torchtitan"
        --pwd /opt/titan-sci
    )

    # Local machine needs cuda-compat workaround
    if [[ "$CLUSTER" == "local" ]]; then
        CMD+=(
            --bind /tmp/cuda-compat:/usr/local/cuda/compat
            --bind /tmp/cuda-lib64:/usr/local/cuda/lib64
            --env LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1
            --env LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/cuda/lib64
        )
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

# Get output directory from config
OUTPUT_DIR=$(run_python "$CONTAINER" "from titan_sci.cluster_config import get_cluster_config; config = get_cluster_config('$CLUSTER', '$TITAN_USER'); print(config['output_dir'])")
[[ -n "$OUTPUT_DIR" ]] || die "Failed to get output_dir for cluster '$CLUSTER'"

export OUTPUT_DIR
echo "Cluster: $CLUSTER | Output: $OUTPUT_DIR | Container: $(basename "$CONTAINER")"

[[ "$DRY_RUN" == false ]] && mkdir -p "$OUTPUT_DIR"

# Build sbatch command: sbatch [slurm-opts] script [training-args]
CMD=(sbatch --output="$OUTPUT_DIR/mpi-out.%j" --error="$OUTPUT_DIR/mpi-err.%j" "${SLURM_OPTS[@]}" "$SLURM_SCRIPT" "${TRAINING_ARGS[@]}")
echo "Command: ${CMD[*]}"
[[ "$DRY_RUN" == true ]] && { echo "[DRY-RUN]"; exit 0; }
exec "${CMD[@]}"
