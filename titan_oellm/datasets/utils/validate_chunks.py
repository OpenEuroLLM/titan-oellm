#!/usr/bin/env python
"""
Diagnostic tool for validating chunked datasets and identifying corrupted chunks.

Usage:
    # Validate all chunks (always validates everything)
    python validate_chunks.py --chunks-dir /path/to/chunks

    # Validate specific chunk(s)
    python validate_chunks.py --chunks-dir /path/to/chunks --chunk-ids 0 10 100

    # Verbose output with detailed diagnostics
    python validate_chunks.py --chunks-dir /path/to/chunks --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from titan_oellm.datasets.dataloader.dataset_validator import (
    DatasetValidator,
    ValidationConfig,
)

logger = logging.getLogger(__name__)


def validate_specific_chunks(
    chunks_dir: str,
    chunk_ids: List[int],
    verbose: bool = False
) -> bool:
    """
    Validate specific chunks by their IDs.

    Args:
        chunks_dir: Directory containing chunk files
        chunk_ids: List of chunk IDs to validate
        verbose: If True, print detailed information

    Returns:
        True if all specified chunks are valid, False otherwise
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return False

    all_valid = True
    for chunk_id in chunk_ids:
        chunk_prefix = chunks_path / f"chunk_{chunk_id:05d}"
        idx_file = Path(f"{chunk_prefix}.idx")
        bin_file = Path(f"{chunk_prefix}.bin")

        if not idx_file.exists() or not bin_file.exists():
            logger.error(f"Chunk {chunk_id}: Files not found")
            all_valid = False
            continue

        logger.info(f"Validating chunk {chunk_id}...")

        # Use the standard validator on this single chunk
        validator = DatasetValidator()
        result = validator.validate_files(str(chunk_prefix))

        if result.is_valid:
            logger.info(f"✓ Chunk {chunk_id}: VALID")
            if verbose:
                print(f"\nChunk {chunk_id} Statistics:")
                for key, value in result.stats.items():
                    print(f"  {key}: {value}")
        else:
            logger.error(f"✗ Chunk {chunk_id}: CORRUPTED")
            all_valid = False
            print(f"\nChunk {chunk_id} Errors:")
            for error in result.errors:
                print(f"  ✗ {error}")
            if result.warnings:
                print(f"\nChunk {chunk_id} Warnings:")
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate chunked datasets and diagnose corruption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--chunks-dir",
        required=True,
        help="Directory containing chunk_*.idx and chunk_*.bin files"
    )

    parser.add_argument(
        "--chunk-ids",
        type=int,
        nargs="+",
        help="Validate specific chunks by ID (e.g., --chunk-ids 0 10 100). If not specified, validates ALL chunks."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics for each validated chunk"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Determine validation mode
    if args.chunk_ids is not None:
        # Validate specific chunks
        logger.info(f"Validating specific chunks: {args.chunk_ids}")
        success = validate_specific_chunks(
            args.chunks_dir,
            args.chunk_ids,
            verbose=args.verbose
        )

    else:
        # Validate all chunks
        logger.info("Validating ALL chunks...")

        # Create validator
        config = ValidationConfig()
        validator = DatasetValidator(config)

        # Run validation
        result = validator.validate_chunked_dataset(chunks_dir=args.chunks_dir)

        # Print results
        print("\n" + "=" * 70)
        print(result)
        print("=" * 70)

        success = result.is_valid

        if args.verbose and result.is_valid:
            print("\nDetailed Statistics:")
            for key, value in result.stats.items():
                print(f"  {key}: {value}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
