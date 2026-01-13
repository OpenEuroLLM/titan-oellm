import logging
import os
import struct
from enum import Enum
from typing import Any, Dict, Optional, Type, Union
import numpy as np
import numpy
import torch

from titan_oellm.datasets.dataloader.dataset_validator import (
    DatasetValidator,
    ValidationConfig,
)

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"

class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class MMapDataset(torch.utils.data.IterableDataset):
    """
    Optimized dataset for random access on large (>1TB) datasets.

    Designed for:
    - Random access with LargeIndexRandomizer
    - Multi-process training without file contention
    - Sub-epoch training on massive datasets
    - No caching/buffering (counterproductive for random access)
    """

    def __init__(
            self,
            path_prefix: str,
            dp_world_size: int,
            dp_rank: int,
            shuffle: bool = True,  # Almost always true for pretraining
            infinite: bool = True,
            limit_samples: Optional[int] = None,
            ignore_samples: Optional[int] = None,
            seed: int = 1,
            validate: bool = True,  # Run validation before training
            validation_config: Optional[ValidationConfig] = None,
            # Validation split support
            exclude_last_n: Optional[int] = None,   # Training: exclude last N samples
            use_only_last_n: Optional[int] = None,  # Validation: use only last N samples
    ) -> None:
        super().__init__()

        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.rng = np.random.RandomState(seed+dp_rank)

        # Store params for checkpointing
        self.limit_samples = limit_samples
        self.ignore_samples = ignore_samples
        self.exclude_last_n = exclude_last_n
        self.use_only_last_n = use_only_last_n

        # Run validation if requested (only on rank 0 to avoid redundant checks)
        if validate and dp_rank == 0:
            logger.info(f"Validating dataset at {path_prefix}...")
            validator = DatasetValidator(validation_config)
            result = validator.validate_files(path_prefix)

            if not result.is_valid:
                error_msg = f"Dataset validation failed:\n{result}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.info(f"Dataset validation passed in {result.elapsed_time:.2f}s")
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(warning)

        # Initialize dataset
        self.initialize(path_prefix, limit_samples, ignore_samples, exclude_last_n, use_only_last_n)

        # Initialize randomizer for this worker
        if shuffle:
            self.random_idx = np.arange(len(self.worker_indices))
            self.rng.shuffle(self.random_idx)

            print(f"Worker {dp_rank}: Shuffle enabled, length {len(self.worker_indices)}")

        self.epoch_counter = 0
        self.sample_counter = 0

    def initialize(self, path_prefix: str, limit_samples: Optional[int], ignore_samples: Optional[int],
                   exclude_last_n: Optional[int] = None, use_only_last_n: Optional[int] = None) -> None:
        """Initialize the dataset with worker-specific data partitioning"""
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"

        assert os.path.exists(idx_path) and os.path.exists(bin_path), \
            f"Files not found at {path_prefix}"

        self.path_prefix = path_prefix

        # Initialize readers optimized for random access
        self.index = _IndexReader(idx_path)
        self.bin_reader = _RandomAccessBinReader(bin_path)

        # Process worker indices with sequential partitioning for better locality
        self.total_samples = len(self.index)
        indices = list(range(self.total_samples))

        # Apply validation split constraints FIRST (before limit/ignore)
        if use_only_last_n is not None and use_only_last_n > 0:
            # Validation mode: use only last N samples
            indices = indices[-use_only_last_n:]
            logger.info(f"Split mode (validation): using last {use_only_last_n} samples")
        elif exclude_last_n is not None and exclude_last_n > 0:
            # Training mode: exclude last N samples
            indices = indices[:-exclude_last_n]
            logger.info(f"Split mode (training): excluding last {exclude_last_n} samples")

        # Then apply limit_samples and ignore_samples
        if limit_samples and isinstance(limit_samples, int):
            indices = indices[:limit_samples]

        if ignore_samples and isinstance(ignore_samples, int):
            indices = indices[ignore_samples:]

        self.total_samples = len(indices)

        # Sequential partitioning for better cache locality in random access
        if self.dp_world_size > 1:
            chunk_size = len(indices) // self.dp_world_size
            start_idx = self.dp_rank * chunk_size

            if self.dp_rank == self.dp_world_size - 1:
                # Last worker gets remainder
                self.worker_indices = indices[start_idx:]
            else:
                self.worker_indices = indices[start_idx:start_idx + chunk_size]
        else:
            self.worker_indices = indices

        if len(self.worker_indices) > 0:
            logger.info(f"Worker {self.dp_rank}: {len(self.worker_indices)} samples "
                        f"(indices {min(self.worker_indices)} to {max(self.worker_indices)})")
        else:
            logger.warning(f"Worker {self.dp_rank}: No samples assigned!")

    def __iter__(self):
        return self

    def __next__(self):
        # Reset randomizer for new epoch
        if self.sample_counter == 0 and self.shuffle:

            self.random_idx = np.arange(len(self.worker_indices))
            self.rng.shuffle(self.random_idx)

            print(">>> Resetting randomizer")

        # Get randomized index
        if self.shuffle:
            idx_in_worker = self.random_idx[self.sample_counter]
            actual_idx = self.worker_indices[idx_in_worker]
        else:
            actual_idx = self.worker_indices[self.sample_counter]

        # Read sequence
        sequence = self.__getitem__(actual_idx)

        # Update counters
        self.sample_counter += 1
        if self.sample_counter >= len(self.worker_indices):
            self.sample_counter = 0
            self.epoch_counter += 1
            if not self.infinite:
                raise StopIteration

        return {'tokens': np.array(sequence)}

    def __getitem__(self, idx: Union[int, numpy.integer]) -> numpy.ndarray:
        """Direct random access - optimized for minimal overhead"""
        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        sequence = self.bin_reader.read(
            dtype=self.index.dtype,
            count=sequence_length,
            offset=sequence_pointer
        )
        return sequence if sequence_mode is None else (sequence, sequence_mode)

    def __len__(self) -> int:
        return len(self.worker_indices)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'worker_rank': self.dp_rank,
            'worker_indices_count': len(self.worker_indices),
            'epoch_counter': self.epoch_counter,
            'sample_counter': self.sample_counter,
            'total_samples': self.total_samples,
        }

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle state without file handles"""
        state = self.__dict__.copy()
        state.pop('bin_reader', None)
        state.pop('index', None)
        state.pop('random_idx', None)  # Will be recreated per process
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and reinitialize file handles"""
        self.__dict__.update(state)
        self.initialize(self.path_prefix,
                        self.limit_samples,
                        self.ignore_samples,
                        self.exclude_last_n,
                        self.use_only_last_n)

        # Recreate random_idx if shuffle is enabled (needed for mid-epoch restore)
        if self.shuffle:
            self.random_idx = np.arange(len(self.worker_indices))
            self.rng.shuffle(self.random_idx)

    def __del__(self) -> None:
        """Clean up resources"""
        if hasattr(self, "bin_reader"):
            del self.bin_reader
        if hasattr(self, "index"):
            del self.index







class _RandomAccessBinReader:
    """
    Binary reader optimized for random access on very large files.

    Key optimizations for >1TB datasets with random access:
    1. Memory mapping with proper flags for OS page sharing
    2. OS hints for random access patterns
    3. Minimal overhead per read
    4. No buffering/caching (counterproductive for random access)
    """

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self.file_size = os.path.getsize(bin_path)

        # Initialize memory mapping optimized for random access
        self._init_mmap_random_access()

        # Set OS hints for random access
        self._set_random_access_hints()

    def _init_mmap_random_access(self):
        """Initialize memory mapping optimized for random access"""
        import mmap

        try:
            # Open with minimal buffering since we're doing random access
            self.file_handle = open(self.bin_path, 'rb', buffering=0)

            # Memory map with flags optimized for random access
            self._mmap = mmap.mmap(
                self.file_handle.fileno(),
                0,  # Map entire file
                access=mmap.ACCESS_READ,
            )

            # Hint that we'll be doing random access
            if hasattr(mmap, 'MADV_RANDOM'):
                try:
                    self._mmap.madvise(mmap.MADV_RANDOM)
                except (OSError, AttributeError):
                    pass

            logger.info(f"Memory mapped {self.file_size / 1024 ** 3:.2f}GB file for random access")

        except Exception as e:
            logger.error(f"Failed to mmap {self.bin_path}: {e}")
            raise

    def _set_random_access_hints(self):
        """Set OS hints for random access patterns"""
        try:
            if hasattr(os, 'posix_fadvise'):
                fd = self.file_handle.fileno()

                # Tell OS we'll be doing random access
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)

                # Don't try to readahead - we're doing random access
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

        except (AttributeError, OSError):
            # Not all systems support these hints
            pass

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """
        Direct random access read with minimal overhead.
        No buffering or caching - each read is independent.
        """
        try:
            dtype_size = np.dtype(dtype).itemsize
            bytes_to_read = count * dtype_size

            # Direct read from memory map - this is as fast as it gets for random access
            data_bytes = self._mmap[offset:offset + bytes_to_read]

            if len(data_bytes) != bytes_to_read:
                raise RuntimeError(f"Could not read {bytes_to_read} bytes at offset {offset}")

            # Zero-copy conversion to numpy array
            return np.frombuffer(data_bytes, dtype=dtype, count=count)

        except Exception as e:
            raise RuntimeError(f"Random access read failed at offset {offset}: {e}")

    def __del__(self):
        """Clean up resources"""
        try:
            if hasattr(self, '_mmap') and self._mmap is not None:
                self._mmap.close()
        except:
            pass

        try:
            if hasattr(self, 'file_handle') and self.file_handle:
                self.file_handle.close()
        except:
            pass


# Keep the original _IndexReader since it's only read once
class _IndexReader:

    def __init__(self, idx_path: str) -> None:
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"Wrong header in {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        # Use memory mapping for index since it's read-heavy
        self.bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        # Extract arrays
        self.sequence_lengths = np.frombuffer(
            self.bin_buffer, dtype=np.int32, count=self.sequence_count, offset=offset
        )

        self.sequence_pointers = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        self.document_indices = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        self.sequence_modes = None

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, idx: int):
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )

    def __del__(self):
        if hasattr(self, 'bin_buffer_mmap'):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

