#!/bin/bash
#
# SLURM Job Submission Wrapper
#
# This script wraps sbatch to dynamically set output/error log paths
# from cluster_paths.toml configuration instead of hardcoded paths.
#
# Usage:
#   bash submit_job.sh [sbatch-options] [slurm-script] [script-args...]
#
# Examples:
#   TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh
#   TITAN_USER=joerg bash submit_job.sh slurm/juwels.sh
#   TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh --nodes=4 --model.flavor=1B
#
# Options:
#   --help      Show this help message
#   --dry-run   Print the sbatch command without executing
#
# Environment variables:
#   TITAN_USER  - Username for user-specific configs (REQUIRED)
#   CLUSTER     - Override cluster detection (default: auto-detect from script name)
#

set -e

# Script directory (for finding container and project files)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# HELP AND OPTIONS
# ============================================================================

show_help() {
    cat << 'EOF'
SLURM Job Submission Wrapper

Usage:
  bash submit_job.sh [sbatch-options] [slurm-script] [script-args...]

Examples:
  TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh
  TITAN_USER=joerg bash submit_job.sh slurm/juwels.sh
  TITAN_USER=joerg CLUSTER=juwels bash submit_job.sh --nodes=4 --model.flavor=1B

Options:
  --help      Show this help message
  --dry-run   Print the sbatch command without executing

Environment variables:
  TITAN_USER  - Username for user-specific configs (REQUIRED)
  CLUSTER     - Cluster name (auto-selects slurm/<CLUSTER>.sh if no script provided)

The wrapper automatically:
  1. Auto-selects slurm script from CLUSTER env var, or detects cluster from script name
  2. Loads output_dir from your user/$TITAN_USER/cluster_paths.toml
  3. Creates output directory if needed
  4. Passes --output and --error to sbatch to override hardcoded paths

EOF
}

# Check for --help or --dry-run
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
    esac
done

# Remove --dry-run from args for sbatch
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--dry-run" ]; then
        ARGS+=("$arg")
    fi
done

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "$TITAN_USER" ]; then
    echo "Error: TITAN_USER environment variable not set." >&2
    echo "Set it before running: export TITAN_USER=your_username" >&2
    echo "See user/example/ for configuration templates." >&2
    exit 1
fi

# Find the slurm script in args (first .sh file)
SLURM_SCRIPT=""
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == *.sh ]]; then
        SLURM_SCRIPT="$arg"
        break
    fi
done

if [ -z "$SLURM_SCRIPT" ]; then
    # Try to auto-select based on CLUSTER env var
    if [ -n "$CLUSTER" ]; then
        SLURM_SCRIPT="slurm/${CLUSTER}.sh"
        if [ ! -f "$SCRIPT_DIR/$SLURM_SCRIPT" ]; then
            echo "Error: Auto-selected script not found: $SLURM_SCRIPT" >&2
            exit 1
        fi
        echo "Auto-selected script: $SLURM_SCRIPT"
        ARGS+=("$SLURM_SCRIPT")
    else
        echo "Error: No SLURM script provided and CLUSTER not set." >&2
        echo "Usage: CLUSTER=juwels bash submit_job.sh" >&2
        echo "   or: bash submit_job.sh slurm/juwels.sh" >&2
        exit 1
    fi
fi

# ============================================================================
# CLUSTER DETECTION
# ============================================================================

# Auto-detect cluster from script name if CLUSTER not set
if [ -z "$CLUSTER" ]; then
    case "$SLURM_SCRIPT" in
        *juwels*)
            CLUSTER="juwels"
            ;;
        *capella*)
            CLUSTER="capella"
            ;;
        *jupiter*)
            CLUSTER="jupiter"
            ;;
        *)
            echo "Error: Cannot auto-detect cluster from script name: $SLURM_SCRIPT" >&2
            echo "Set CLUSTER environment variable explicitly." >&2
            exit 1
            ;;
    esac
fi

echo "Detected cluster: $CLUSTER"

# ============================================================================
# FIND CONTAINER
# ============================================================================

# Find the container file (pattern: titan_*.sif)
CONTAINER=$(find "$SCRIPT_DIR" -maxdepth 1 -name "titan_*.sif" -type f | head -n1)

if [ -z "$CONTAINER" ]; then
    echo "Error: No container file (titan_*.sif) found in $SCRIPT_DIR" >&2
    exit 1
fi

echo "Using container: $(basename "$CONTAINER")"

# ============================================================================
# LOAD OUTPUT DIRECTORY FROM CLUSTER CONFIG
# ============================================================================

echo "Loading output directory from cluster config..."

# Run inside container to access cluster_config module
OUTPUT_DIR=$(apptainer exec \
    --env TITAN_USER="$TITAN_USER" \
    --bind "$SCRIPT_DIR":/opt/titan-oellm \
    "$CONTAINER" \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-oellm')
from titan_oellm.cluster_config import get_submit_config
try:
    config = get_submit_config('$CLUSTER')
    print(config['output_dir'])
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Failed to get output_dir from cluster config" >&2
    echo "Please ensure user/$TITAN_USER/cluster_paths.toml has [cluster.$CLUSTER] section" >&2
    exit 1
fi

echo "Output directory: $OUTPUT_DIR"

# Export for downstream jobs (passed through sbatch environment)
export OUTPUT_DIR

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/slurm"
    echo "Created output directory (if not exists)"
fi

# ============================================================================
# BUILD AND EXECUTE SBATCH COMMAND
# ============================================================================

# Build the sbatch command with dynamic output/error paths
SBATCH_CMD=(
    sbatch
    --output="$OUTPUT_DIR/slurm/mpi-out.%j"
    --error="$OUTPUT_DIR/slurm/mpi-err.%j"
    "${ARGS[@]}"
)

echo ""
echo "Command: ${SBATCH_CMD[*]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would execute the above command"
else
    exec "${SBATCH_CMD[@]}"
fi
