#!/usr/bin/env python
"""
Standalone script to validate bin/idx dataset files for corruption and quality issues.

Usage:

PROJECT_DIR=$(pwd)                    
CONTAINER="titan_juwels_0.2.0.sif"  
DATASET=/p/data1/.../neox_neomotron_cc_v1
ml Apptainer-Tools

# Define variables
apptainer exec --nv \
    --pwd /opt/titan-sci \
    --bind $DATASET:$DATASET \
    --bind $PROJECT_DIR:/opt/titan-sci \
    $CONTAINER \
    python -m titan_oellm.datasets.validate_dataset --path-prefix $DATASET
    python -m titan_oellm.datasets.validate_dataset --path-prefix /path/to/dataset --vocab-size 50257 --verbose
"""

import argparse
import sys
from pathlib import Path

from titan_oellm.datasets.dataloader.dataset_validator import (
    DatasetValidator,
    ValidationConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate bin/idx dataset files for corruption and quality issues"
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        required=True,
        help="Path prefix for .bin and .idx files (without extension)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for token range validation (optional)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random samples to validate (default: 10)",
    )
    parser.add_argument(
        "--min-diversity",
        type=float,
        default=0.1,
        help="Minimum token diversity threshold (default: 0.1)",
    )
    parser.add_argument(
        "--max-token-freq",
        type=float,
        default=0.8,
        help="Maximum single token frequency threshold (default: 0.8)",
    )
    parser.add_argument(
        "--max-run-length",
        type=int,
        default=50,
        help="Maximum consecutive identical token run length (default: 50)",
    )
    parser.add_argument(
        "--no-first-last",
        action="store_true",
        help="Skip checking first and last samples",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate path exists
    bin_path = Path(f"{args.path_prefix}.bin")
    idx_path = Path(f"{args.path_prefix}.idx")

    if not bin_path.exists():
        print(f"Error: Binary file not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    if not idx_path.exists():
        print(f"Error: Index file not found: {idx_path}", file=sys.stderr)
        sys.exit(1)

    # Configure validation
    config = ValidationConfig(
        num_random_samples=args.num_samples,
        check_first_last=not args.no_first_last,
        min_token_diversity=args.min_diversity,
        max_token_frequency=args.max_token_freq,
        max_token_run_length=args.max_run_length,
        vocab_size=args.vocab_size,
    )

    if args.verbose:
        print(f"Validating dataset: {args.path_prefix}")
        print(f"Configuration:")
        print(f"  - Random samples: {config.num_random_samples}")
        print(f"  - Check first/last: {config.check_first_last}")
        print(f"  - Min diversity: {config.min_token_diversity:.2%}")
        print(f"  - Max token freq: {config.max_token_frequency:.2%}")
        print(f"  - Max run length: {config.max_token_run_length}")
        if config.vocab_size:
            print(f"  - Vocab size: {config.vocab_size}")
        print()

    # Run validation
    validator = DatasetValidator(config)
    result = validator.validate_files(args.path_prefix)

    # Print results
    print(result)
    print()

    # Exit with appropriate code
    if result.is_valid:
        print("✓ Dataset validation PASSED")
        sys.exit(0)
    else:
        print("✗ Dataset validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
