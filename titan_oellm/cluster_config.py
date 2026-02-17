"""
Cluster configuration management for Titan-OELLM.

This module provides automatic cluster detection and path resolution for
datasets and tokenizers across different HPC clusters (JUWELS, Jupiter, Capella).

Dataset naming convention: dataset.{dataset_name}.{tokenizer_name}.{cluster}
This explicitly encodes which tokenizer was used for pre-tokenization.

Usage:
    # Auto-detect cluster and get paths
    from titan_oellm.cluster_config import get_cli_args
    args = get_cli_args('slimpajama_627b', 'neox')

    # List available configurations
    from titan_oellm.cluster_config import list_available
    list_available()

    # Manually specify cluster
    args = get_cli_args('slimpajama_627b', 'neox', cluster='juwels')
"""

import os
import socket
import sys
import tomli
from pathlib import Path
from typing import Optional


def detect_cluster() -> str:
    """
    Auto-detect cluster from hostname or CLUSTER environment variable.

    Returns:
        str: Cluster name ('local', 'juwels', 'jupiter', or 'capella')

    Raises:
        ValueError: If hostname doesn't match any known cluster and CLUSTER env var not set
    """
    hostname = socket.gethostname().lower()

    # JUWELS cluster detection
    if "jwlogin" in hostname or "jwc" in hostname or "juwels" in hostname:
        return "juwels"

    # Jupiter cluster detection
    elif "jupiter" in hostname or "jrc" in hostname:
        return "jupiter"

    # Capella cluster detection (hostname is c1, c2, etc.)
    elif hostname.startswith("c") and len(hostname) >= 2 and hostname[1].isdigit():
        return "capella"
    elif "capella" in hostname:
        return "capella"

    elif "leonardo" in hostname:
        return "leonardo"

    else:
        raise ValueError(
            f"Unknown cluster for hostname: {hostname}\n"
            f"Expected hostname to contain: jwlogin/jwc/juwels (JUWELS), "
            f"jupiter/jrc (Jupiter), capella (Capella) or leonardo"
        )


def load_cluster_paths(user: Optional[str] = None) -> dict:
    """
    Load cluster_paths.toml configuration file from user-specific directory.

    Args:
        user: Username to load config for (requires TITAN_USER env var if not specified)

    Returns:
        dict: Parsed TOML configuration

    Raises:
        ValueError: If TITAN_USER environment variable is not set
    """
    # Determine user from parameter or environment variable
    if user is None:
        user = os.environ.get('TITAN_USER')
        if user is None:
            raise ValueError(
                "TITAN_USER environment variable not set.\n"
                "Set it to your username: export TITAN_USER=your_username\n"
                "See user/example/ for configuration templates."
            )

    # Construct path to user-specific cluster_paths.toml
    project_root = Path(__file__).parent.parent
    config_path = project_root / "user" / user / "cluster_paths.toml"

    # Fallback to old location with deprecation warning
    if not config_path.exists():
        old_config_path = Path(__file__).parent / "configs" / "cluster_paths.toml"
        if old_config_path.exists():
            print(
                f"Warning: Using deprecated config location: {old_config_path}\n"
                f"Please move cluster_paths.toml to: {config_path}",
                file=sys.stderr,
            )
            config_path = old_config_path
        else:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure cluster_paths.toml exists in user/{user}/ directory."
            )

    with open(config_path, "rb") as f:
        return tomli.load(f)


def get_tokenizer_path(tokenizer: str, cluster: Optional[str] = None, user: Optional[str] = None) -> str:
    """
    Get tokenizer path for specified cluster.

    Args:
        tokenizer: Tokenizer name (e.g., 'neox', 'nemotron')
        cluster: Cluster name (auto-detected if None)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        str: Absolute path to tokenizer directory

    Raises:
        ValueError: If tokenizer not found for cluster
    """
    if cluster is None:
        cluster = detect_cluster()

    config = load_cluster_paths(user=user)

    # Lookup tokenizer path
    tokenizer_key = f"tokenizer.{tokenizer}.{cluster}"
    if tokenizer_key not in config:
        available = [k for k in config.keys() if k.startswith(f"tokenizer.{tokenizer}.")]
        available_clusters = [k.split('.')[-1] for k in available]
        raise ValueError(
            f"Tokenizer '{tokenizer}' not found for cluster '{cluster}'.\n"
            f"Available clusters for this tokenizer: {', '.join(available_clusters) or 'none'}\n"
            f"Available tokenizers: {', '.join(_extract_names('tokenizer', config))}"
        )

    return config[tokenizer_key]["path"]


def get_benchmark_paths(
    tokenizer: str, cluster: Optional[str] = None, user: Optional[str] = None, validate: bool = True
) -> dict:
    """
    Get benchmark paths for specified tokenizer and cluster.

    Args:
        tokenizer: Tokenizer name (e.g., 'neox', 'nemotron')
        cluster: Cluster name (auto-detected if None)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')
        validate: Whether to validate that benchmark files exist (default: True)

    Returns:
        dict: Dictionary with benchmark path prefixes:
            - wikitext2_path: Path prefix for WikiText-2 (without .bin/.idx)
            - wikitext103_path: Path prefix for WikiText-103
            - lambada_path: Path prefix for LAMBADA

    Raises:
        ValueError: If benchmarks not found for tokenizer/cluster
        FileNotFoundError: If validate=True and benchmark files don't exist
    """
    if cluster is None:
        cluster = detect_cluster()

    config = load_cluster_paths(user=user)

    # Lookup benchmark base path
    benchmark_key = f"benchmarks.{tokenizer}.{cluster}"
    if benchmark_key not in config:
        available = [k for k in config.keys() if k.startswith(f"benchmarks.{tokenizer}.")]
        available_clusters = [k.split(".")[-1] for k in available]
        raise ValueError(
            f"Benchmarks for tokenizer '{tokenizer}' not found for cluster '{cluster}'.\n"
            f"Available clusters for this tokenizer: {', '.join(available_clusters) or 'none'}\n"
            f"Available benchmark configurations: {', '.join(_extract_names('benchmarks', config))}"
        )

    base_path = config[benchmark_key]["path"]

    # Construct full paths for each benchmark
    # Structure: {base_path}/wikitext2/wikitext2.{bin,idx}
    paths = {
        "wikitext2_path": f"{base_path}/wikitext2/wikitext2",
        "wikitext103_path": f"{base_path}/wikitext103/wikitext103",
        "lambada_path": f"{base_path}/lambada/lambada",
    }

    # Validate that benchmark files exist, use empty placeholder if missing
    if validate:
        missing = []
        for name, path_prefix in paths.items():
            bin_path = Path(f"{path_prefix}.bin")
            idx_path = Path(f"{path_prefix}.idx")
            if not bin_path.exists() or not idx_path.exists():
                missing.append(name)
                paths[name] = ""  # Use empty placeholder

        if missing:
            print(
                f"Warning: Benchmark files not found for {', '.join(missing)}. "
                f"Benchmarks will be skipped. Run scripts/download_benchmarks.py to enable them.",
                file=sys.stderr
            )

    return paths


def get_paths(dataset: str, tokenizer: str, cluster: Optional[str] = None, user: Optional[str] = None) -> dict:
    """
    Get resolved paths for dataset and tokenizer on specified cluster.

    Args:
        dataset: Dataset name (e.g., 'slimpajama_627b', 'fineweb_edu')
        tokenizer: Tokenizer name (e.g., 'neox', 'llama3')
        cluster: Cluster name (auto-detected if None)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        dict: Dictionary with resolved paths:
            - cluster: Detected/specified cluster name
            - tokenizer_path: Absolute path to tokenizer
            - data_prefix: Training data prefix path
            - chunks_dir: Training data chunks directory
            - validation_prefix: Validation data prefix path
            - dataloader: Dataloader type
            - min_doc_len: Minimum document length

    Raises:
        ValueError: If dataset or tokenizer not found for cluster
    """
    if cluster is None:
        cluster = detect_cluster()

    config = load_cluster_paths(user=user)

    # Lookup tokenizer path
    tokenizer_key = f"tokenizer.{tokenizer}.{cluster}"
    if tokenizer_key not in config:
        available = [k for k in config.keys() if k.startswith(f"tokenizer.{tokenizer}.")]
        available_clusters = [k.split(".")[-1] for k in available]
        raise ValueError(
            f"Tokenizer '{tokenizer}' not found for cluster '{cluster}'.\n"
            f"Available clusters for this tokenizer: {', '.join(available_clusters) or 'none'}\n"
            f"Available tokenizers: {', '.join(_extract_names('tokenizer', config))}"
        )

    tokenizer_path = config[tokenizer_key]["path"]

    # Lookup dataset configuration
    dataset_key = f"dataset.{dataset}.{tokenizer}.{cluster}"
    if dataset_key not in config:
        available = [k for k in config.keys() if k.startswith(f"dataset.{dataset}.{tokenizer}.")]
        available_clusters = [k.split(".")[-1] for k in available]
        raise ValueError(
            f"Dataset '{dataset}' with tokenizer '{tokenizer}' not found for cluster '{cluster}'.\n"
            f"Available clusters for this dataset-tokenizer pair: {', '.join(available_clusters) or 'none'}\n"
            f"Available dataset-tokenizer pairs: {', '.join(_extract_names('dataset', config))}"
        )

    dataset_config = config[dataset_key]

    cfg = {
        "cluster": cluster,
        "tokenizer_path": tokenizer_path,
        "data_prefix": dataset_config["train_prefix"],
        "chunks_dir": dataset_config["train_chunks"],
        "validation_prefix": dataset_config["validation_prefix"],
        "dataloader": dataset_config["dataloader"],
        "min_doc_len": dataset_config["min_doc_len"],
    }
    return cfg


def get_cli_args(
    dataset: str = "slimpajama_627b",
    tokenizer: str = "neox",
    cluster: Optional[str] = None,
    config_file: str = 'base_norm.toml',
    config_base_path: str = '/opt/titan-oellm/titan_oellm/configs',
    validate: bool = True,
    user: Optional[str] = None,
) -> str:
    """
    Generate CLI arguments string for training script.

    Args:
        dataset: Dataset name (default: 'slimpajama_627b')
        tokenizer: Tokenizer name (default: 'neox')
        cluster: Cluster name (auto-detected if None)
        config_file: Config filename for validation (default: 'base_norm.toml')
        config_base_path: Base path to config directory (default: '/opt/titan-oellm/titan_oellm/configs')
        validate: Whether to validate paths before returning (default: True)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        str: Space-separated CLI arguments for training script

    Raises:
        RuntimeError: If validation fails (when validate=True)

    Example:
        >>> args = get_cli_args('slimpajama_627b', 'neox', 'juwels', 'base_norm.toml')
        >>> # Returns: "--model.tokenizer_path=... --data.data_prefix=... ..."
    """
    # Validate paths if requested
    if validate:
        if cluster is None:
            cluster = detect_cluster()

        valid, messages = validate_paths(dataset, tokenizer, cluster, config_file, config_base_path, user=user)

        # Print all messages (errors and warnings)
        for msg in messages:
            print(msg, file=sys.stderr)

        if not valid:
            raise RuntimeError("Path validation failed. See errors above.")

    paths = get_paths(dataset, tokenizer, cluster, user=user)

    # Get benchmark paths (validation handled by get_benchmark_paths)
    benchmark_paths = get_benchmark_paths(tokenizer, cluster, user=user, validate=validate)

    return _format_cli_args(paths, benchmark_paths)


def _format_cli_args(paths: dict, benchmark_paths: dict) -> str:
    """Format resolved paths into CLI args string."""
    return (
        f"--model.tokenizer_path={paths['tokenizer_path']} "
        f"--data.data_prefix={paths['data_prefix']} "
        f"--data.chunks_dir={paths['chunks_dir']} "
        f"--data.dataloader={paths['dataloader']} "
        f"--data.min_doc_len={paths['min_doc_len']} "
        f"--validation.data_prefix={paths['validation_prefix']} "
        f"--benchmarks.wikitext2_path={benchmark_paths['wikitext2_path']} "
        f"--benchmarks.wikitext103_path={benchmark_paths['wikitext103_path']} "
        f"--benchmarks.lambada_path={benchmark_paths['lambada_path']}"
    )


def validate_paths(
    dataset: str,
    tokenizer: str,
    cluster: str,
    config_file: str,
    config_base_path: str = "/opt/titan-oellm/titan_oellm/configs",
    user: Optional[str] = None
) -> tuple[bool, list[str]]:
    """
    Validate that all required paths exist and are not empty.

    Args:
        dataset: Dataset name (e.g., 'slimpajama_627b')
        tokenizer: Tokenizer name (e.g., 'neox')
        cluster: Cluster name (e.g., 'juwels')
        config_file: Config filename (e.g., 'base_norm.toml')
        config_base_path: Base path to config directory
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        tuple[bool, list[str]]: (all_valid, messages)
            - all_valid: True if all validations passed
            - messages: List of error or warning messages

    Example:
        >>> valid, errors = validate_paths('slimpajama_627b', 'neox', 'juwels', 'base_norm.toml')
        >>> if not valid:
        ...     for error in errors:
        ...         print(error)
    """
    from pathlib import Path

    messages = []
    all_valid = True

    # Check config file exists
    config_path = Path(config_base_path) / config_file
    if not config_path.exists():
        messages.append(f"Error: Config file not found: {config_path}")
        all_valid = False
    elif not config_path.is_file():
        messages.append(f"Error: Config path is not a file: {config_path}")
        all_valid = False

    # Get paths from cluster_config
    try:
        paths = get_paths(dataset, tokenizer, cluster, user=user)
    except ValueError as e:
        messages.append(f"Error: Failed to resolve paths: {e}")
        return False, messages

    # Check tokenizer directory exists and is not empty
    tokenizer_path = Path(paths["tokenizer_path"])
    try:
        # Use .resolve(strict=True) to force exceptions on access issues
        resolved_path = tokenizer_path.resolve(strict=True)
        # Check if it's a directory by trying to iterate (will raise if not a directory)
        try:
            if not any(resolved_path.iterdir()):
                messages.append(f"Error: Tokenizer directory is empty: {tokenizer_path}")
                all_valid = False
        except NotADirectoryError:
            messages.append(f"Error: Tokenizer path is not a directory: {tokenizer_path}")
            all_valid = False
    except FileNotFoundError:
        messages.append(f"Error: Tokenizer directory not found: {tokenizer_path}")
        all_valid = False
    except PermissionError as e:
        messages.append(f"Error: Permission denied accessing tokenizer directory: {tokenizer_path} - {e}")
        all_valid = False
    except OSError as e:
        messages.append(f"Error: OS error accessing tokenizer directory: {tokenizer_path} - {e}")
        all_valid = False

    # Check training data prefix exists
    data_prefix = paths["data_prefix"]
    # For mmap datasets, check if any files with the prefix exist
    data_prefix_path = Path(data_prefix)
    parent_dir = data_prefix_path.parent
    prefix_name = data_prefix_path.name

    try:
        # Use .resolve(strict=True) to force exceptions on access issues
        resolved_parent = parent_dir.resolve(strict=True)
        # Try to glob to check for permission issues
        matching_files = list(resolved_parent.glob(f"{prefix_name}*"))
        if not matching_files:
            messages.append(f"Error: No training data files found with prefix: {data_prefix}")
            all_valid = False
    except FileNotFoundError:
        messages.append(f"Error: Training data directory not found: {parent_dir}")
        all_valid = False
    except PermissionError as e:
        messages.append(f"Error: Permission denied accessing training data directory: {parent_dir} - {e}")
        all_valid = False
    except OSError as e:
        messages.append(f"Error: OS error accessing training data directory: {parent_dir} - {e}")
        all_valid = False

    # Check chunks directory if using ChunkedMMapDataset
    if paths["dataloader"] == "ChunkedMMapDataset":
        chunks_dir = Path(paths["chunks_dir"])
        try:
            # Use .resolve(strict=True) to force exceptions on access issues
            resolved_chunks = chunks_dir.resolve(strict=True)
            # Check if it's a directory by trying to iterate (will raise if not a directory)
            try:
                if not any(resolved_chunks.iterdir()):
                    messages.append(f"Error: Chunks directory is empty: {chunks_dir}")
                    all_valid = False
            except NotADirectoryError:
                messages.append(f"Error: Chunks path is not a directory: {chunks_dir}")
                all_valid = False
        except FileNotFoundError:
            messages.append(f"Error: Chunks directory not found: {chunks_dir}")
            all_valid = False
        except PermissionError as e:
            messages.append(f"Error: Permission denied accessing chunks directory: {chunks_dir} - {e}")
            all_valid = False
        except OSError as e:
            messages.append(f"Error: OS error accessing chunks directory: {chunks_dir} - {e}")
            all_valid = False

    # Check validation data prefix exists
    validation_prefix = paths["validation_prefix"]
    validation_prefix_path = Path(validation_prefix)
    validation_parent = validation_prefix_path.parent
    validation_name = validation_prefix_path.name

    try:
        # Use .resolve(strict=True) to force exceptions on access issues
        resolved_validation = validation_parent.resolve(strict=True)
        # Try to glob to check for permission issues
        matching_files = list(resolved_validation.glob(f"{validation_name}*"))
        if not matching_files:
            messages.append(f"Warning: No validation data files found with prefix: {validation_prefix}")
    except FileNotFoundError:
        messages.append(f"Warning: Validation data directory not found: {validation_parent}")
        # Validation is optional, so don't set all_valid = False
    except PermissionError as e:
        messages.append(f"Warning: Permission denied accessing validation data directory: {validation_parent} - {e}")
        # Validation is optional, so don't set all_valid = False
    except OSError as e:
        messages.append(f"Warning: OS error accessing validation data directory: {validation_parent} - {e}")
        # Validation is optional, so don't set all_valid = False

    # Add success message if everything is valid
    if all_valid and not any(msg.startswith("Warning") for msg in messages):
        messages.append("All paths validated successfully")

    return all_valid, messages


def _extract_names(prefix: str, config: dict) -> list[str]:
    """
    Extract unique names from config keys with given prefix.

    For datasets: returns 'dataset.tokenizer' pairs (e.g., 'nemotron_cc.nemotron')
    For tokenizers: returns tokenizer names (e.g., 'neox', 'nemotron')
    For benchmarks: returns 'tokenizer.cluster' pairs (e.g., 'neox.juwels')
    """
    names = set()
    for key in config.keys():
        if key.startswith(f"{prefix}."):
            parts = key.split(".")
            if prefix == "dataset" and len(parts) >= 3:
                # Format: dataset.name.tokenizer.cluster -> return 'name.tokenizer'
                names.add(f"{parts[1]}.{parts[2]}")
            elif prefix == "tokenizer" and len(parts) >= 2:
                # Format: tokenizer.name.cluster -> return 'name'
                names.add(parts[1])
            elif prefix == "benchmarks" and len(parts) >= 3:
                # Format: benchmarks.tokenizer.cluster -> return 'tokenizer.cluster'
                names.add(f"{parts[1]}.{parts[2]}")
    return sorted(names)


def list_available(user: Optional[str] = None) -> None:
    """
    Print all available configurations (clusters, datasets, tokenizers).

    Args:
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Example output:
        Available configurations:
          Clusters: capella, jupiter, juwels
          Dataset-tokenizer pairs: nemotron_cc.nemotron, slimpajama_627b.neox
          Tokenizers: neox, nemotron
    """
    config = load_cluster_paths(user=user)

    datasets = _extract_names("dataset", config)
    tokenizers = _extract_names("tokenizer", config)

    # Extract unique clusters
    clusters = set()
    for key in config.keys():
        parts = key.split(".")
        # Handle both tokenizer.name.cluster (3 parts) and dataset.name.tokenizer.cluster (4 parts)
        if len(parts) == 3 and key.startswith("tokenizer."):
            clusters.add(parts[2])
        elif len(parts) == 4 and key.startswith("dataset."):
            clusters.add(parts[3])

    print("Available configurations:")
    print(f"  Clusters: {', '.join(sorted(clusters))}")
    print(f"  Dataset-tokenizer pairs: {', '.join(datasets)}")
    print(f"  Tokenizers: {', '.join(tokenizers)}")


def get_cluster_config(cluster: Optional[str] = None, user: Optional[str] = None) -> dict:
    """
    Get cluster-specific configuration (paths, cache directories, container name).

    Args:
        cluster: Cluster name (auto-detected if None)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        dict: Dictionary with cluster configuration:
            - output_dir: Path to output/logs directory
            - cache_base: Base path for cache directories
            - triton_cache: Path to Triton cache (auto-generated from cache_base)
            - hf_datasets_cache: Path to HuggingFace datasets cache
            - hf_home: Path to HuggingFace home
            - torch_home: Path to PyTorch cache
            - apptainer_cachedir: Path to Apptainer cache directory
            - apptainer_tmpdir: Path to Apptainer temp directory
            - data_dir: Optional data directory (Capella-specific)

    Raises:
        ValueError: If cluster configuration not found

    Example:
        >>> config = get_cluster_config('juwels')
        >>> print(config['triton_cache'])
    """
    if cluster is None:
        cluster = detect_cluster()

    config = load_cluster_paths(user=user)

    # Lookup cluster configuration
    cluster_key = f"cluster.{cluster}"
    if cluster_key not in config:
        available_clusters = [k.split(".")[1] for k in config.keys() if k.startswith("cluster.")]
        raise ValueError(
            f"Cluster '{cluster}' configuration not found.\n"
            f"Available clusters: {', '.join(available_clusters) or 'none'}\n"
            f"Please add [cluster.{cluster}] section to your cluster_paths.toml"
        )

    cluster_config = config[cluster_key]

    # Auto-generate cache paths from cache_base
    cache_base = cluster_config.get('cache_base', '')

    result = {
        'output_dir': cluster_config.get('output_dir', ''),
        'cache_base': cache_base,
        # Auto-generated from cache_base, but can be overridden
        'triton_cache': cluster_config.get('triton_cache', f"{cache_base}/triton"),
        'hf_datasets_cache': cluster_config.get('hf_datasets_cache', f"{cache_base}/hf"),
        'hf_home': cluster_config.get('hf_home', f"{cache_base}/hf"),
        'torch_home': cluster_config.get('torch_home', f"{cache_base}/torch"),
        'apptainer_cachedir': cluster_config.get('apptainer_cachedir', f"{cache_base}/apptainer"),
        'apptainer_tmpdir': cluster_config.get('apptainer_tmpdir', f"{cache_base}/apptainer"),
    }

    # Add optional data_dir if present (Capella-specific)
    if 'data_dir' in cluster_config:
        result['data_dir'] = cluster_config['data_dir']

    return result


def get_env_exports(cluster: Optional[str] = None, user: Optional[str] = None) -> str:
    """
    Generate shell export statements for cluster-specific cache directories.

    Args:
        cluster: Cluster name (auto-detected if None)
        user: Username for config lookup (defaults to TITAN_USER env var, then 'joerg')

    Returns:
        str: Shell export statements for cache environment variables

    Example:
        >>> exports = get_env_exports('juwels')
        >>> print(exports)
        ...
    """
    config = get_cluster_config(cluster, user)

    exports = []
    # Triton cache for kernel compilation
    exports.append(f'export TRITON_CACHE_DIR="{config["triton_cache"]}"')

    # PyTorch Inductor cache (torch.compile) - CRITICAL for cache persistence!
    # Uses same directory as TRITON_CACHE_DIR for compiled artifacts
    exports.append(f'export TORCHINDUCTOR_CACHE_DIR="{config["triton_cache"]}"')

    # Enable FX graph caching - required for torch.compile cache to work
    exports.append('export TORCHINDUCTOR_FX_GRAPH_CACHE="1"')

    # Enable AOTAutograd caching for additional speedup
    exports.append('export TORCHINDUCTOR_AUTOGRAD_CACHE="1"')

    # Disable CUDA graphs to save memory (they use extra memory during recording)
    # This prevents OOM when using high batch sizes with torch.compile
    # Need to disable BOTH cudagraphs and cudagraph_trees!
    exports.append('export TORCHINDUCTOR_TRITON_CUDAGRAPHS="False"')
    exports.append('export TORCHINDUCTOR_TRITON_CUDAGRAPH_TREES="False"')

    # Other cache directories
    exports.append(f'export HF_DATASETS_CACHE="{config["hf_datasets_cache"]}"')
    exports.append(f'export HF_HOME="{config["hf_home"]}"')
    exports.append(f'export TORCH_HOME="{config["torch_home"]}"')
    exports.append(f'export APPTAINER_CACHEDIR="{config["apptainer_cachedir"]}"')
    exports.append(f'export APPTAINER_TMPDIR="{config["apptainer_tmpdir"]}"')

    # Add DATA_DIR if present (Capella-specific)
    if 'data_dir' in config:
        exports.append(f'export DATA_DIR="{config["data_dir"]}"')

    return "\n".join(exports)


if __name__ == "__main__":
    # Command-line interface for testing
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_available()
    elif len(sys.argv) > 1 and sys.argv[1] == "detect":
        try:
            cluster = detect_cluster()
            print(f"Detected cluster: {cluster}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Usage: python cluster_config.py validate <dataset> <tokenizer> <cluster> <config_file> [config_base_path]
        if len(sys.argv) < 6:
            print(
                "Usage: python cluster_config.py validate <dataset> <tokenizer> <cluster> <config_file> [config_base_path]"
            )
            print(
                "Example: python cluster_config.py validate slimpajama_627b neox juwels base_norm.toml /opt/titan-oellm/titan_oellm/configs"
            )
            sys.exit(1)

        dataset = sys.argv[2]
        tokenizer = sys.argv[3]
        cluster = sys.argv[4]
        config_file = sys.argv[5]
        config_base_path = sys.argv[6] if len(sys.argv) > 6 else "/opt/titan-oellm/titan_oellm/configs"

        valid, messages = validate_paths(dataset, tokenizer, cluster, config_file, config_base_path)
        for msg in messages:
            print(msg)
        sys.exit(0 if valid else 1)
    elif len(sys.argv) >= 3:
        dataset = sys.argv[1]
        tokenizer = sys.argv[2]
        cluster = sys.argv[3] if len(sys.argv) > 3 else None
        try:
            args = get_cli_args(dataset, tokenizer, cluster)
            print(args)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:")
        print("  python cluster_config.py list")
        print("  python cluster_config.py detect")
        print("  python cluster_config.py validate <dataset> <tokenizer> <cluster> <config_file> [config_base_path]")
        print("  python cluster_config.py <dataset> <tokenizer> [cluster]")
        print()
        print("Examples:")
        print("  python cluster_config.py list")
        print("  python cluster_config.py slimpajama_627b neox")
        print("  python cluster_config.py slimpajama_627b neox juwels")
        print("  python cluster_config.py validate slimpajama_627b neox juwels base_norm.toml")
