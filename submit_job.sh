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
  bash submit_job.sh [options] [slurm-script] [training-args...]
  bash submit_job.sh --local [training-args...]

Options:
  --help, -h   Show this help
  --dry-run    Print command without executing
  --local      Run locally without SLURM

Environment:
  TITAN_USER   Username for configs (REQUIRED)
  CLUSTER      Cluster name (default: auto-detect or 'local')
  DATASET      Dataset name (default: test_dataset)
  TOKENIZER    Tokenizer name (default: neox)
  CONFIG       Config file path for --local mode
  NPROC        Number of GPUs for local (default: 1)

Examples:
  TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh --nodes=4
  TITAN_USER=joerg bash submit_job.sh --local --model.flavor=debugmodel
EOF
}

find_container() {
    local container
    container=$(find "$SCRIPT_DIR" -maxdepth 1 -name "titan_*.sif" -type f | head -n1)
    [[ -n "$container" ]] || die "No container (titan_*.sif) found in $SCRIPT_DIR"
    echo "$container"
}

run_python() {
    local container=$1 code=$2
    apptainer exec --env TITAN_USER="$TITAN_USER" --bind "$SCRIPT_DIR":/opt/titan-oellm \
        "$container" python3 -c "
import sys; sys.path.insert(0, '/opt/titan-oellm')
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
    CLUSTER="${CLUSTER:-local}"
    DATASET="${DATASET:-test_dataset}"
    TOKENIZER="${TOKENIZER:-neox}"
    NPROC="${NPROC:-1}"
    CONFIG="${CONFIG:-user/$TITAN_USER/configs/debug.toml}"
    CONTAINER=$(find_container)
    
    # Format config file path for container
    CONFIG_ARG="--job.config_file=/opt/titan-oellm/$CONFIG"
    
    echo "=== Local Training: cluster=$CLUSTER dataset=$DATASET tokenizer=$TOKENIZER gpus=$NPROC ==="
    
    # Load environment from cluster config
    eval "$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_env_exports; print(get_env_exports('$CLUSTER'))")"
    [[ -n "${TRITON_CACHE_DIR:-}" ]] || die "Failed to load cluster config for '$CLUSTER'"
    
    # Get dataset/tokenizer CLI args
    CLUSTER_ARGS=$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_dataset_args; print(get_dataset_args('$DATASET', '$TOKENIZER', '$CLUSTER'))")
    
    # Create directories
    mkdir -p "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$OUTPUT_DIR"

    # Load CUDA modules for JUWELS (required for --nv flag to work properly)
    if [[ "$CLUSTER" == "juwels" ]]; then
        echo "Loading JUWELS CUDA modules..."
        ml Stages/2025
        ml GCC/13.3.0
        ml Python/3.12.3
        ml CUDA/12
        ml cuDNN/9.5.0.50-CUDA-12
        ml NCCL/default-CUDA-12
        ml NVHPC/25.5-CUDA-12
    fi

    # Build command
    CMD=(
        apptainer exec --nv
        --bind "$TRITON_CACHE_DIR:$TRITON_CACHE_DIR"
        --bind "$SCRIPT_DIR:/opt/titan-oellm"
        --bind "$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"
        --bind "$TORCH_HOME:$TORCH_HOME"
        --bind "$HOME:$HOME"
        --env TITAN_USER="$TITAN_USER"
        --env OUTPUT_DIR="$OUTPUT_DIR"
        --env TORCH_HOME="$TORCH_HOME"
        --pwd /opt/titan-oellm
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

# Find slurm script in args or auto-select
SLURM_SCRIPT=""
for arg in "${ARGS[@]}"; do
    [[ "$arg" == *.sh ]] && SLURM_SCRIPT="$arg" && break
done

if [[ -z "$SLURM_SCRIPT" ]]; then
    [[ -n "${CLUSTER:-}" ]] || die "No SLURM script provided and CLUSTER not set"
    SLURM_SCRIPT="slurm/${CLUSTER}.sh"
    [[ -f "$SCRIPT_DIR/$SLURM_SCRIPT" ]] || die "Auto-selected script not found: $SLURM_SCRIPT"
    ARGS+=("$SLURM_SCRIPT")
fi

# Detect cluster from script name if not set
[[ -z "${CLUSTER:-}" ]] && CLUSTER=$(detect_cluster_from_script "$SLURM_SCRIPT")

CONTAINER=$(find_container)

# Get output directory from config
OUTPUT_DIR=$(run_python "$CONTAINER" "from titan_oellm.cluster_config import get_submit_config; print(get_submit_config('$CLUSTER')['output_dir'])")
[[ -n "$OUTPUT_DIR" ]] || die "Failed to get output_dir for cluster '$CLUSTER'"

export OUTPUT_DIR
echo "Cluster: $CLUSTER | Output: $OUTPUT_DIR | Container: $(basename "$CONTAINER")"

[[ "$DRY_RUN" == false ]] && mkdir -p "$OUTPUT_DIR/slurm"

CMD=(sbatch --output="$OUTPUT_DIR/slurm/mpi-out.%j" --error="$OUTPUT_DIR/slurm/mpi-err.%j" "${ARGS[@]}")
echo "Command: ${CMD[*]}"
[[ "$DRY_RUN" == true ]] && { echo "[DRY-RUN]"; exit 0; }
exec "${CMD[@]}"
