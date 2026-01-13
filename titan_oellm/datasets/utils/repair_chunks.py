#!/usr/bin/env python
"""
Repair tool for corrupted chunk files.

Options:
1. Delete corrupted chunks (safest, loses ~0.2% of data)
2. Salvage by truncating index to match binary file (risky, may lose sequences)
3. List corrupted chunks for manual recreation from source

Usage:
    # Identify corrupted chunks
    python repair_chunks.py --chunks-dir /path/to/chunks --identify

    # Delete corrupted chunks
    python repair_chunks.py --chunks-dir /path/to/chunks --delete

    # Attempt to salvage by truncating index
    python repair_chunks.py --chunks-dir /path/to/chunks --salvage

    # Dry run (show what would be done without doing it)
    python repair_chunks.py --chunks-dir /path/to/chunks --delete --dry-run
"""

import argparse
import logging
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def find_corrupted_chunks(chunks_dir: str) -> List[Tuple[str, int, int, int]]:
    """
    Find all corrupted chunks in the directory.

    Returns:
        List of tuples: (chunk_prefix, required_bytes, actual_bytes, shortfall)
    """
    from titan_oellm.datasets.dataloader.mmap_dataset_chunked import (
        _IndexReader,
        _RandomAccessBinReader,
    )

    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    idx_files = list(chunks_path.glob("chunk_*.idx"))
    chunk_prefixes = []

    for idx_file in idx_files:
        chunk_prefix = str(idx_file).replace('.idx', '')
        bin_file = Path(f"{chunk_prefix}.bin")
        if bin_file.exists():
            chunk_prefixes.append(chunk_prefix)

    corrupted = []

    for chunk_prefix in sorted(chunk_prefixes):
        try:
            idx_path = f"{chunk_prefix}.idx"
            bin_path = f"{chunk_prefix}.bin"

            index = _IndexReader(idx_path)
            bin_reader = _RandomAccessBinReader(bin_path)

            if len(index) == 0:
                continue

            # Calculate maximum offset required
            dtype_size = np.dtype(index.dtype).itemsize
            max_offsets = index.sequence_pointers + index.sequence_lengths * dtype_size
            max_required = np.max(max_offsets)

            if max_required > bin_reader.file_size:
                shortfall = max_required - bin_reader.file_size
                corrupted.append((
                    chunk_prefix,
                    max_required,
                    bin_reader.file_size,
                    shortfall
                ))
                logger.info(f"Found corrupted: {Path(chunk_prefix).name}")

        except Exception as e:
            logger.error(f"Error checking {chunk_prefix}: {e}")

    return corrupted


def delete_corrupted_chunks(
    corrupted: List[Tuple[str, int, int, int]],
    dry_run: bool = False
) -> None:
    """
    Delete corrupted chunk files.

    Args:
        corrupted: List of corrupted chunk info from find_corrupted_chunks
        dry_run: If True, only show what would be deleted
    """
    print("\n" + "=" * 70)
    print("DELETE CORRUPTED CHUNKS")
    print("=" * 70)

    if not corrupted:
        print("No corrupted chunks to delete.")
        return

    print(f"\nFound {len(corrupted)} corrupted chunk(s):")
    for chunk_prefix, required, actual, shortfall in corrupted:
        chunk_name = Path(chunk_prefix).name
        print(f"  - {chunk_name}: shortfall {shortfall:,} bytes ({shortfall/1024/1024:.2f} MB)")

    if dry_run:
        print("\n[DRY RUN] Would delete:")
        for chunk_prefix, _, _, _ in corrupted:
            print(f"  - {chunk_prefix}.idx")
            print(f"  - {chunk_prefix}.bin")
        print("\nRe-run without --dry-run to actually delete.")
        return

    print("\nDeleting corrupted chunks...")
    for chunk_prefix, _, _, _ in corrupted:
        idx_path = Path(f"{chunk_prefix}.idx")
        bin_path = Path(f"{chunk_prefix}.bin")

        if idx_path.exists():
            idx_path.unlink()
            logger.info(f"Deleted {idx_path}")

        if bin_path.exists():
            bin_path.unlink()
            logger.info(f"Deleted {bin_path}")

    print(f"✓ Deleted {len(corrupted)} corrupted chunk(s)")
    print("\nNote: The dataloader will skip these missing chunks automatically.")


def salvage_chunk(chunk_prefix: str, dry_run: bool = False) -> bool:
    """
    Attempt to salvage a corrupted chunk by truncating the index.

    This finds the last sequence that fits completely in the binary file
    and creates a new truncated index.

    Args:
        chunk_prefix: Path prefix for the chunk
        dry_run: If True, only show what would be done

    Returns:
        True if salvage was successful, False otherwise
    """
    from titan_oellm.datasets.dataloader.mmap_dataset_chunked import _IndexReader

    try:
        idx_path = f"{chunk_prefix}.idx"
        bin_path = f"{chunk_prefix}.bin"

        # Load index and get binary file size
        index = _IndexReader(idx_path)
        bin_size = os.path.getsize(bin_path)
        dtype_size = np.dtype(index.dtype).itemsize

        # Find last valid sequence
        valid_count = 0
        for i in range(len(index)):
            pointer, length, _ = index[i]
            end_offset = pointer + (length * dtype_size)

            if end_offset <= bin_size:
                valid_count = i + 1
            else:
                break

        if valid_count == 0:
            logger.error(f"Cannot salvage {Path(chunk_prefix).name}: No valid sequences")
            return False

        lost_sequences = len(index) - valid_count
        logger.info(f"Can salvage {valid_count}/{len(index)} sequences "
                   f"(losing {lost_sequences} sequences)")

        if dry_run:
            return True

        # Create backup
        backup_idx = f"{idx_path}.backup"
        shutil.copy2(idx_path, backup_idx)
        logger.info(f"Created backup: {backup_idx}")

        # Write truncated index
        with open(idx_path, 'rb') as f_in:
            # Read header
            header = f_in.read(9)
            version = struct.unpack("<Q", f_in.read(8))[0]
            dtype_code = struct.unpack("<B", f_in.read(1))[0]
            sequence_count = struct.unpack("<Q", f_in.read(8))[0]
            document_count = struct.unpack("<Q", f_in.read(8))[0]

            # Read arrays
            offset = f_in.tell()
            sequence_lengths = np.frombuffer(
                f_in.read(sequence_count * 4),
                dtype=np.int32
            )
            sequence_pointers = np.frombuffer(
                f_in.read(sequence_count * 8),
                dtype=np.int64
            )
            document_indices = np.frombuffer(
                f_in.read(document_count * 8),
                dtype=np.int64
            )

        # Truncate to valid sequences
        sequence_lengths_truncated = sequence_lengths[:valid_count]
        sequence_pointers_truncated = sequence_pointers[:valid_count]

        # Adjust document indices
        # Keep only document boundaries that are within valid range
        document_indices_truncated = document_indices[document_indices < valid_count]

        # Write new index
        with open(idx_path, 'wb') as f_out:
            # Write header
            f_out.write(header)
            f_out.write(struct.pack("<Q", version))
            f_out.write(struct.pack("<B", dtype_code))
            f_out.write(struct.pack("<Q", valid_count))
            f_out.write(struct.pack("<Q", len(document_indices_truncated)))

            # Write arrays
            sequence_lengths_truncated.tofile(f_out)
            sequence_pointers_truncated.tofile(f_out)
            document_indices_truncated.tofile(f_out)

        logger.info(f"✓ Salvaged {Path(chunk_prefix).name}: "
                   f"{valid_count} sequences retained, {lost_sequences} lost")
        return True

    except Exception as e:
        logger.error(f"Failed to salvage {Path(chunk_prefix).name}: {e}")
        return False


def salvage_corrupted_chunks(
    corrupted: List[Tuple[str, int, int, int]],
    dry_run: bool = False
) -> None:
    """
    Attempt to salvage all corrupted chunks.

    Args:
        corrupted: List of corrupted chunk info from find_corrupted_chunks
        dry_run: If True, only show what would be done
    """
    print("\n" + "=" * 70)
    print("SALVAGE CORRUPTED CHUNKS")
    print("=" * 70)
    print("\n⚠ WARNING: This will modify chunk files!")
    print("         Backups will be created with .backup extension")

    if not corrupted:
        print("\nNo corrupted chunks to salvage.")
        return

    if dry_run:
        print("\n[DRY RUN] Would attempt to salvage:")
    else:
        print("\nAttempting to salvage:")

    success_count = 0
    for chunk_prefix, required, actual, shortfall in corrupted:
        chunk_name = Path(chunk_prefix).name
        print(f"\n  {chunk_name}:")
        print(f"    Required: {required:,} bytes")
        print(f"    Actual:   {actual:,} bytes")
        print(f"    Shortfall: {shortfall:,} bytes ({shortfall/1024/1024:.2f} MB)")

        if salvage_chunk(chunk_prefix, dry_run):
            success_count += 1
            if not dry_run:
                print(f"    ✓ Salvaged")
        else:
            print(f"    ✗ Could not salvage")

    if dry_run:
        print(f"\n[DRY RUN] Would salvage {success_count}/{len(corrupted)} chunks")
        print("\nRe-run without --dry-run to actually salvage.")
    else:
        print(f"\n✓ Salvaged {success_count}/{len(corrupted)} chunks")
        if success_count < len(corrupted):
            print(f"  Failed to salvage {len(corrupted) - success_count} chunks")
            print("  Consider deleting these with --delete")


def main():
    parser = argparse.ArgumentParser(
        description="Repair corrupted chunk files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--chunks-dir",
        required=True,
        help="Directory containing chunk files"
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--identify",
        action="store_true",
        help="Only identify corrupted chunks without taking action"
    )
    action_group.add_argument(
        "--delete",
        action="store_true",
        help="Delete corrupted chunks (safest option)"
    )
    action_group.add_argument(
        "--salvage",
        action="store_true",
        help="Attempt to salvage by truncating index (creates backups)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
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

    print("Scanning for corrupted chunks...")
    corrupted = find_corrupted_chunks(args.chunks_dir)

    if not corrupted:
        print("\n✓ No corrupted chunks found!")
        return 0

    print(f"\nFound {len(corrupted)} corrupted chunk(s):")
    for chunk_prefix, required, actual, shortfall in corrupted:
        chunk_name = Path(chunk_prefix).name
        print(f"  {chunk_name}:")
        print(f"    Required: {required:,} bytes ({required/1024/1024:.2f} MB)")
        print(f"    Actual:   {actual:,} bytes ({actual/1024/1024:.2f} MB)")
        print(f"    Shortfall: {shortfall:,} bytes ({shortfall/1024/1024:.2f} MB)")

    if args.identify:
        print("\nTo repair, run one of:")
        print(f"  # Delete corrupted chunks (recommended):")
        print(f"  python repair_chunks.py --chunks-dir {args.chunks_dir} --delete")
        print(f"\n  # Attempt to salvage (risky):")
        print(f"  python repair_chunks.py --chunks-dir {args.chunks_dir} --salvage")
        return 0

    if args.delete:
        delete_corrupted_chunks(corrupted, dry_run=args.dry_run)

    elif args.salvage:
        salvage_corrupted_chunks(corrupted, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
