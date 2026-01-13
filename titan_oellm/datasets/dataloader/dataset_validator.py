"""
Dataset validator for bin/idx files.

Provides fast, comprehensive validation to detect:
- File corruption and truncation
- Invalid pointers and inaccessible samples
- Data quality issues (low diversity, pathological patterns)
- Vocabulary violations
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import numpy

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for dataset validation"""
    # Sampling strategy
    num_random_samples: int = 10
    check_first_last: bool = True

    # Quality thresholds
    min_token_diversity: float = 0.1  # 10% unique tokens
    max_token_frequency: float = 0.8   # No token >80%
    max_token_run_length: int = 50     # Max consecutive same token

    # Vocabulary validation
    vocab_size: Optional[int] = None   # If provided, check token ranges

    # Performance
    max_validation_time: float = 5.0   # seconds


@dataclass
class ValidationResult:
    """Result of dataset validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]
    elapsed_time: float

    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ CORRUPTED"
        lines = [
            "Dataset Validation Report",
            "=" * 50,
            f"Status: {status}",
            f"Time elapsed: {self.elapsed_time:.2f}s",
            ""
        ]

        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  ✗ {err}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")
            lines.append("")

        if self.stats:
            lines.append("Statistics:")
            for key, value in self.stats.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)


class DatasetValidator:
    """
    Fast multi-level validation for bin/idx datasets.

    Performs comprehensive validation in under 5 seconds even for TB-scale datasets:
    - Level 1: File structure and bounds checking
    - Level 2: Sample accessibility verification
    - Level 3: Content quality validation (diversity, patterns, vocabulary)

    Usage:
        # Standalone validation
        validator = DatasetValidator()
        result = validator.validate_files("/path/to/dataset")
        if not result.is_valid:
            print(result)
            raise RuntimeError("Dataset is corrupted")

        # Custom configuration
        config = ValidationConfig(
            num_random_samples=20,
            min_token_diversity=0.15,
            vocab_size=50257
        )
        validator = DatasetValidator(config)
        result = validator.validate_files("/path/to/dataset")

        # Automatic validation in dataset (enabled by default)
        from titan_oellm.datasets.dataloader.mmap_dataset import MMapDataset
        dataset = MMapDataset(
            path_prefix="/path/to/dataset",
            dp_rank=0,
            dp_world_size=1,
            validate=True  # Will raise error if corrupted
        )
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def validate_files(self, path_prefix: str) -> ValidationResult:
        """
        Perform fast multi-level validation on bin/idx files.

        Args:
            path_prefix: Path prefix for .bin and .idx files

        Returns:
            ValidationResult with status, errors, warnings, and stats
        """
        start_time = time.time()
        self.errors = []
        self.warnings = []
        self.stats = {}

        try:
            # Level 1: File structure validation
            bin_path, idx_path = self._validate_file_structure(path_prefix)

            # Level 2: Bounds validation
            index, bin_reader = self._validate_bounds(idx_path, bin_path)

            # Level 3: Content quality validation
            self._validate_content_quality(index, bin_reader)

        except Exception as e:
            self.errors.append(f"Validation failed: {str(e)}")

        elapsed_time = time.time() - start_time
        is_valid = len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats,
            elapsed_time=elapsed_time
        )

    def _validate_file_structure(self, path_prefix: str) -> Tuple[str, str]:
        """Level 1: Validate file existence and basic structure"""
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"

        # Check file existence
        if not os.path.exists(idx_path):
            self.errors.append(f"Index file not found: {idx_path}")
        if not os.path.exists(bin_path):
            self.errors.append(f"Binary file not found: {bin_path}")

        if self.errors:
            raise RuntimeError("File existence check failed")

        # Get file sizes
        bin_size = os.path.getsize(bin_path)
        idx_size = os.path.getsize(idx_path)

        self.stats["bin_file_size_gb"] = f"{bin_size / 1024**3:.2f} GB"
        self.stats["idx_file_size_mb"] = f"{idx_size / 1024**2:.2f} MB"

        return bin_path, idx_path

    def _validate_bounds(self, idx_path: str, bin_path: str) -> Tuple[Any, Any]:
        """Level 2: Validate that all pointers are within bin file bounds"""
        # Import here to avoid circular dependency
        from titan_oellm.datasets.dataloader.mmap_dataset import (
            _IndexReader,
            _RandomAccessBinReader,
        )

        # Load index
        index = _IndexReader(idx_path)
        bin_reader = _RandomAccessBinReader(bin_path)

        # Check if index is valid
        if len(index) == 0:
            self.errors.append("Index contains no samples")
            raise RuntimeError("Empty index")

        self.stats["total_samples"] = len(index)
        self.stats["total_documents"] = index.document_count

        # Calculate maximum offset required
        dtype_size = np.dtype(index.dtype).itemsize
        max_pointer = np.max(index.sequence_pointers)
        max_length = np.max(index.sequence_lengths)

        # Find the actual last byte needed
        last_sample_idx = np.argmax(index.sequence_pointers + index.sequence_lengths * dtype_size)
        last_byte_offset = (index.sequence_pointers[last_sample_idx] +
                           index.sequence_lengths[last_sample_idx] * dtype_size)

        self.stats["max_offset_gb"] = f"{last_byte_offset / 1024**3:.2f} GB"
        self.stats["max_sample_length"] = int(max_length)
        self.stats["dtype"] = index.dtype.__name__

        # Validate bounds
        if last_byte_offset > bin_reader.file_size:
            self.errors.append(
                f"Index points beyond bin file: last required byte at {last_byte_offset}, "
                f"but bin file is only {bin_reader.file_size} bytes"
            )
            raise RuntimeError("Bounds check failed")

        # Try reading first and last samples
        sample_indices = []
        if self.config.check_first_last:
            sample_indices.extend([0, len(index) - 1])

        for idx in sample_indices:
            try:
                pointer, length, _ = index[idx]
                _ = bin_reader.read(dtype=index.dtype, count=length, offset=pointer)
            except Exception as e:
                self.errors.append(f"Failed to read sample {idx}: {str(e)}")

        return index, bin_reader

    def _validate_content_quality(self, index: Any, bin_reader: Any):
        """Level 3: Validate content quality of sampled sequences"""
        # Select samples to validate
        total_samples = len(index)
        sample_indices = []

        if self.config.check_first_last:
            sample_indices.extend([0, total_samples - 1])

        # Add random samples
        if self.config.num_random_samples > 0:
            # Stratified sampling across the dataset
            step = max(1, total_samples // (self.config.num_random_samples + 2))
            random_indices = list(range(step, total_samples - 1, step))[:self.config.num_random_samples]
            sample_indices.extend(random_indices)

        # Remove duplicates and sort
        sample_indices = sorted(set(sample_indices))

        # Quality metrics
        diversities = []
        max_frequencies = []
        max_run_lengths = []
        vocab_violations = []

        for idx in sample_indices:
            try:
                pointer, length, _ = index[idx]
                tokens = bin_reader.read(dtype=index.dtype, count=length, offset=pointer)

                # Check token diversity
                unique_tokens = len(np.unique(tokens))
                diversity = unique_tokens / len(tokens) if len(tokens) > 0 else 0
                diversities.append(diversity)

                if diversity < self.config.min_token_diversity:
                    self.warnings.append(
                        f"Sample {idx}: Low token diversity ({diversity:.2%}, "
                        f"threshold: {self.config.min_token_diversity:.2%})"
                    )

                # Check max token frequency
                if len(tokens) > 0:
                    token_counts = np.bincount(tokens.astype(np.int64))
                    max_freq = np.max(token_counts) / len(tokens)
                    max_frequencies.append(max_freq)

                    if max_freq > self.config.max_token_frequency:
                        self.warnings.append(
                            f"Sample {idx}: High token dominance ({max_freq:.2%}, "
                            f"threshold: {self.config.max_token_frequency:.2%})"
                        )

                # Check for long runs of identical tokens
                if len(tokens) > 1:
                    max_run = self._find_max_run_length(tokens)
                    max_run_lengths.append(max_run)

                    if max_run > self.config.max_token_run_length:
                        self.errors.append(
                            f"Sample {idx}: Pathological token pattern detected "
                            f"({max_run} consecutive identical tokens, "
                            f"threshold: {self.config.max_token_run_length})"
                        )

                # Check vocabulary range
                if self.config.vocab_size is not None:
                    out_of_range = np.sum((tokens < 0) | (tokens >= self.config.vocab_size))
                    if out_of_range > 0:
                        vocab_violations.append(idx)
                        self.errors.append(
                            f"Sample {idx}: {out_of_range} tokens out of vocabulary range "
                            f"[0, {self.config.vocab_size})"
                        )

            except Exception as e:
                self.errors.append(f"Failed to validate sample {idx}: {str(e)}")

        # Aggregate statistics
        if diversities:
            self.stats["avg_token_diversity"] = f"{np.mean(diversities):.2%}"
            self.stats["min_token_diversity"] = f"{np.min(diversities):.2%}"

        if max_frequencies:
            self.stats["avg_max_token_freq"] = f"{np.mean(max_frequencies):.2%}"
            self.stats["max_token_freq"] = f"{np.max(max_frequencies):.2%}"

        if max_run_lengths:
            self.stats["avg_max_run_length"] = f"{np.mean(max_run_lengths):.1f}"
            self.stats["max_run_length"] = int(np.max(max_run_lengths))

        self.stats["samples_validated"] = len(sample_indices)

        if vocab_violations:
            self.stats["samples_with_vocab_violations"] = len(vocab_violations)

    @staticmethod
    def _find_max_run_length(tokens: np.ndarray) -> int:
        """Find the maximum run length of consecutive identical tokens"""
        if len(tokens) == 0:
            return 0

        max_run = 1
        current_run = 1

        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        return max_run

    def validate_chunked_dataset(
        self,
        chunks_dir: str
    ) -> ValidationResult:
        """
        Validate a chunked dataset by checking ALL chunk files.

        Args:
            chunks_dir: Directory containing chunk_*.idx and chunk_*.bin files

        Returns:
            ValidationResult with aggregated status across all chunks
        """
        start_time = time.time()
        self.errors = []
        self.warnings = []
        self.stats = {}

        from pathlib import Path

        chunks_path = Path(chunks_dir)
        if not chunks_path.exists():
            self.errors.append(f"Chunks directory not found: {chunks_dir}")
            return ValidationResult(
                is_valid=False,
                errors=self.errors,
                warnings=self.warnings,
                stats=self.stats,
                elapsed_time=time.time() - start_time
            )

        # Discover all chunk files
        idx_files = list(chunks_path.glob("chunk_*.idx"))
        chunk_prefixes = []

        for idx_file in idx_files:
            chunk_prefix = str(idx_file).replace('.idx', '')
            bin_file = Path(f"{chunk_prefix}.bin")

            if bin_file.exists():
                chunk_prefixes.append(chunk_prefix)
            else:
                self.warnings.append(f"Missing .bin file for {idx_file}")

        chunk_prefixes.sort()
        total_chunks = len(chunk_prefixes)

        if total_chunks == 0:
            self.errors.append(f"No valid chunk pairs found in {chunks_dir}")
            return ValidationResult(
                is_valid=False,
                errors=self.errors,
                warnings=self.warnings,
                stats=self.stats,
                elapsed_time=time.time() - start_time
            )

        self.stats["total_chunks"] = total_chunks
        self.stats["validation_mode"] = "ALL chunks"

        # Validate all chunks
        chunks_to_validate = chunk_prefixes
        self.stats["chunks_validated"] = len(chunks_to_validate)

        # Validate each selected chunk
        chunk_results = []
        total_samples = 0
        corrupted_chunks = []

        for i, chunk_prefix in enumerate(chunks_to_validate):
            chunk_name = Path(chunk_prefix).name
            logger.debug(f"Validating chunk {i+1}/{len(chunks_to_validate)}: {chunk_name}")

            try:
                # Validate this chunk using existing single-file validation logic
                idx_path = f"{chunk_prefix}.idx"
                bin_path = f"{chunk_prefix}.bin"

                # Load and validate bounds for this chunk
                index, _ = self._validate_bounds_for_chunk(
                    idx_path, bin_path, chunk_name
                )

                if index is not None:
                    total_samples += len(index)
                    chunk_results.append({
                        'chunk': chunk_name,
                        'samples': len(index),
                        'status': 'valid'
                    })

            except Exception as e:
                corrupted_chunks.append(chunk_name)
                self.errors.append(f"Chunk {chunk_name}: {str(e)}")
                chunk_results.append({
                    'chunk': chunk_name,
                    'status': 'corrupted',
                    'error': str(e)
                })

        # Aggregate statistics
        self.stats["total_samples_in_validated_chunks"] = total_samples
        self.stats["corrupted_chunks"] = len(corrupted_chunks)
        self.stats["valid_chunks"] = len(chunks_to_validate) - len(corrupted_chunks)

        if corrupted_chunks:
            self.stats["corrupted_chunk_list"] = ", ".join(corrupted_chunks[:5])
            if len(corrupted_chunks) > 5:
                self.stats["corrupted_chunk_list"] += f" ... and {len(corrupted_chunks) - 5} more"

        elapsed_time = time.time() - start_time
        is_valid = len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats,
            elapsed_time=elapsed_time
        )

    def _validate_bounds_for_chunk(
        self,
        idx_path: str,
        bin_path: str,
        chunk_name: str
    ) -> Tuple[Any, Any]:
        """
        Validate bounds for a single chunk file.

        Similar to _validate_bounds but with chunk-specific error messages.
        """
        # Import here to avoid circular dependency
        from titan_oellm.datasets.dataloader.mmap_dataset_chunked import (
            _IndexReader,
            _RandomAccessBinReader,
        )

        # Load index
        index = _IndexReader(idx_path)
        bin_reader = _RandomAccessBinReader(bin_path)

        # Check if index is valid
        if len(index) == 0:
            self.warnings.append(f"Chunk {chunk_name}: Empty (0 samples)")
            return None, None

        # Calculate maximum offset required
        dtype_size = np.dtype(index.dtype).itemsize

        # Find the actual last byte needed
        last_sample_idx = np.argmax(index.sequence_pointers + index.sequence_lengths * dtype_size)
        last_byte_offset = (index.sequence_pointers[last_sample_idx] +
                           index.sequence_lengths[last_sample_idx] * dtype_size)

        # Validate bounds
        if last_byte_offset > bin_reader.file_size:
            raise RuntimeError(
                f"Index requires {last_byte_offset} bytes, but binary file is only "
                f"{bin_reader.file_size} bytes (shortfall: {last_byte_offset - bin_reader.file_size} bytes)"
            )

        # Try reading first and last samples
        if self.config.check_first_last:
            for idx in [0, len(index) - 1]:
                try:
                    pointer, length, _ = index[idx]
                    _ = bin_reader.read(dtype=index.dtype, count=length, offset=pointer)
                except Exception as e:
                    raise RuntimeError(f"Failed to read sample {idx}: {str(e)}")

        return index, bin_reader
