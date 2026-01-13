import logging
import os
import struct
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type, Union, List
import numpy as np
import numpy
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


# Import the existing DType class and readers (keeping the same as original)
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


class ChunkedMMapDataset(torch.utils.data.IterableDataset):
    """
    Optimized dataset for chunked data with worker-specific chunk assignment.

    Key features:
    - Each worker gets assigned multiple chunks randomly (seeded)
    - Reads documents consecutively within each chunk (pre-shuffled)
    - Efficient chunk transitions
    - Load balancing across workers
    """

    def __init__(
            self,
            chunks_dir: str,
            dp_world_size: int,
            dp_rank: int,
            infinite: bool = True,
            seed: int = 1,
            # Per-chunk split support for validation
            exclude_first_n_per_chunk: Optional[int] = None,     # Training: skip first N docs in each chunk
            use_only_first_n_per_chunk: Optional[int] = None, # Validation: use only first N docs in each chunk
    ) -> None:
        super().__init__()

        self.chunks_dir = Path(chunks_dir)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.infinite = infinite
        self.seed = seed

        # Store split params for checkpointing
        self.exclude_first_n_per_chunk = exclude_first_n_per_chunk
        self.use_only_first_n_per_chunk = use_only_first_n_per_chunk

        # Discover and assign chunks
        self.all_chunks = self.get_chunk_prefixes()
        logger.info(f"Found {len(self.all_chunks)} total chunks in {self.chunks_dir}")
        self.assign_chunks_to_worker(self.all_chunks)

        # Initialize state
        self.current_chunk_idx = 0
        self.current_chunk_position = 0
        self.epoch_counter = 0
        self.sample_counter = 0

        # Effective chunk boundaries (set by load_current_chunk)
        self.effective_chunk_start = 0
        self.effective_chunk_end = 0

        # Load first chunk
        self.load_current_chunk()

        # Log split mode info
        split_info = ""
        if exclude_first_n_per_chunk:
            split_info = f" (training mode: skip first {exclude_first_n_per_chunk} docs/chunk)"
        elif use_only_first_n_per_chunk:
            split_info = f" (validation mode: use only first {use_only_first_n_per_chunk} docs/chunk)"

        logger.info(f"Worker {dp_rank}: Assigned {len(self.worker_chunks)} chunks, "
                    f"total samples: {self.total_worker_samples}{split_info}")

    def get_chunk_prefixes(self) -> None:
        """Discover all available chunk files"""
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {self.chunks_dir}")

        # Find all .idx files and extract chunk paths
        idx_files = list(self.chunks_dir.glob("chunk_*.idx"))
        all_chunks = []

        for idx_file in idx_files:
            chunk_prefix = str(idx_file).replace('.idx', '')
            bin_file = Path(f"{chunk_prefix}.bin")

            if bin_file.exists():
                all_chunks.append(chunk_prefix)
            else:
                logger.warning(f"Missing .bin file for {idx_file}")

        all_chunks.sort()  # Ensure consistent ordering
        logger.info(f"Discovered {len(all_chunks)} chunks")

        if not all_chunks:
            raise RuntimeError(f"No valid chunk pairs found in {self.chunks_dir}")
        return all_chunks

    def assign_chunks_to_worker(self, all_chunks: List) -> None:
        """Randomly assign chunks to this worker"""
        total_chunks = len(all_chunks)

        # Create a shuffled list of chunk indices
        chunk_indices = list(range(total_chunks))

        if self.dp_world_size > 1:
            worker_chunk_indices = [chunk_indices[i] for i in range(total_chunks) if i % self.dp_world_size == self.dp_rank]
        else:
            worker_chunk_indices = chunk_indices

        self.worker_chunks = [all_chunks[i] for i in worker_chunk_indices]

        # Load chunk metadata to calculate total samples
        self.chunk_lengths = []
        self.effective_chunk_lengths = []  # Lengths after split
        self.total_worker_samples = 0

        for chunk_path in self.worker_chunks:
            try:
                idx_reader = _IndexReader(f"{chunk_path}.idx")
                chunk_length = len(idx_reader)
                self.chunk_lengths.append(chunk_length)

                # Calculate effective length based on split mode
                effective_length = self._calculate_effective_length(chunk_length)
                self.effective_chunk_lengths.append(effective_length)
                self.total_worker_samples += effective_length

                del idx_reader  # Clean up
            except Exception as e:
                logger.error(f"Failed to read chunk {chunk_path}: {e}")
                raise

        logger.info(f"Dataloader Worker {self.dp_rank}: Chunks {len(self.worker_chunks)} assignment with {self.total_worker_samples} total documents")

    def _calculate_effective_length(self, chunk_length: int) -> int:
        """Calculate effective chunk length after applying split constraints"""
        if self.use_only_first_n_per_chunk is not None:
            # Validation: use only first N docs
            return min(self.use_only_first_n_per_chunk, chunk_length)
        elif self.exclude_first_n_per_chunk is not None:
            # Training: skip first N docs
            return max(0, chunk_length - self.exclude_first_n_per_chunk)
        else:
            # No split: use entire chunk
            return chunk_length

    def load_current_chunk(self) -> None:
        """Load the current chunk for reading"""
        if self.current_chunk_idx >= len(self.worker_chunks):
            if self.infinite:
                # Reset to first chunk for new epoch
                self.current_chunk_idx = 0
                self.epoch_counter += 1
                rng = np.random.RandomState(self.seed + self.epoch_counter)
                self.worker_chunks = rng.permutation(self.worker_chunks).tolist()
                logger.info(f"Worker {self.dp_rank}: Starting epoch {self.epoch_counter}")
            else:
                raise StopIteration

        chunk_path = self.worker_chunks[self.current_chunk_idx]

        # Clean up previous chunk if exists
        if hasattr(self, 'current_index'):
            del self.current_index
        if hasattr(self, 'current_bin_reader'):
            del self.current_bin_reader

        # Load new chunk
        self.current_index = _IndexReader(f"{chunk_path}.idx")
        self.current_bin_reader = _RandomAccessBinReader(f"{chunk_path}.bin")
        self.current_chunk_length = len(self.current_index)

        # Apply per-chunk split to set effective boundaries
        if self.use_only_first_n_per_chunk is not None:
            # Validation: only use first N docs
            self.effective_chunk_start = 0
            self.effective_chunk_end = min(self.use_only_first_n_per_chunk, self.current_chunk_length)
        elif self.exclude_first_n_per_chunk is not None:
            # Training: skip first N docs
            self.effective_chunk_start = min(self.exclude_first_n_per_chunk, self.current_chunk_length)
            self.effective_chunk_end = self.current_chunk_length
        else:
            # No split: use entire chunk
            self.effective_chunk_start = 0
            self.effective_chunk_end = self.current_chunk_length

        # Set position to effective start
        self.current_chunk_position = self.effective_chunk_start

        # Validate chunk integrity
        self._validate_chunk_bounds(chunk_path)

        logger.debug(f"Worker {self.dp_rank}: Loaded chunk {self.current_chunk_idx} "
                     f"({chunk_path}) with {self.current_chunk_length} samples "
                     f"(effective range: {self.effective_chunk_start}-{self.effective_chunk_end})")

    def _validate_chunk_bounds(self, chunk_path: str) -> None:
        """
        Validate that all sequences in the index can be read from the binary file.

        This catches corrupted or truncated chunks at load time instead of during training.
        """
        if self.current_chunk_length == 0:
            return  # Empty chunk is valid

        # Calculate the maximum byte offset required
        dtype_size = np.dtype(self.current_index.dtype).itemsize
        max_offset = 0
        max_offset_idx = 0

        for i in range(self.current_chunk_length):
            pointer, length, _ = self.current_index[i]
            end_offset = pointer + (length * dtype_size)
            if end_offset > max_offset:
                max_offset = end_offset
                max_offset_idx = i

        # Check if the binary file is large enough
        bin_file_size = self.current_bin_reader.file_size
        if max_offset > bin_file_size:
            pointer, length, _ = self.current_index[max_offset_idx]
            raise RuntimeError(
                f"Chunk validation failed: {chunk_path}\n"
                f"  Index requires {max_offset} bytes, but binary file is only {bin_file_size} bytes\n"
                f"  Shortfall: {max_offset - bin_file_size} bytes\n"
                f"  Problematic sequence: index {max_offset_idx}, "
                f"offset {pointer}, length {length}, dtype_size {dtype_size}\n"
                f"  This usually indicates the chunk was corrupted or incompletely written during creation."
            )

    def __iter__(self):
        return self

    def __next__(self):
        # Check if we need to load next chunk (using effective_chunk_end)
        if self.current_chunk_position >= self.effective_chunk_end:
            self.current_chunk_idx += 1
            self.load_current_chunk()

        # Read sequence from current position in current chunk
        sequence = self.read_sequence_at_position(self.current_chunk_position)

        # Update position
        self.current_chunk_position += 1
        self.sample_counter += 1

        return {'tokens': np.array(sequence)}

    def read_sequence_at_position(self, position: int) -> numpy.ndarray:
        """Read sequence at specific position in current chunk"""
        sequence_pointer, sequence_length, sequence_mode = self.current_index[position]
        sequence = self.current_bin_reader.read(
            dtype=self.current_index.dtype,
            count=sequence_length,
            offset=sequence_pointer
        )
        return sequence if sequence_mode is None else (sequence, sequence_mode)

    def __len__(self) -> int:
        """Total samples across all assigned chunks"""
        return self.total_worker_samples

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'worker_rank': self.dp_rank,
            'assigned_chunks': len(self.worker_chunks),
            'current_chunk_idx': self.current_chunk_idx,
            'current_chunk_position': self.current_chunk_position,
            'epoch_counter': self.epoch_counter,
            'sample_counter': self.sample_counter,
            'total_worker_samples': self.total_worker_samples,
            'chunk_lengths': self.chunk_lengths,
            'effective_chunk_lengths': self.effective_chunk_lengths,
            'exclude_first_n_per_chunk': self.exclude_first_n_per_chunk,
            'use_only_first_n_per_chunk': self.use_only_first_n_per_chunk,
        }

    def get_current_chunk_info(self) -> Dict[str, Any]:
        """Get information about currently loaded chunk"""
        if hasattr(self, 'current_index'):
            effective_length = self.effective_chunk_end - self.effective_chunk_start
            effective_position = self.current_chunk_position - self.effective_chunk_start
            return {
                'chunk_idx': self.current_chunk_idx,
                'chunk_path': self.worker_chunks[self.current_chunk_idx],
                'chunk_length': self.current_chunk_length,
                'effective_chunk_start': self.effective_chunk_start,
                'effective_chunk_end': self.effective_chunk_end,
                'effective_length': effective_length,
                'position': self.current_chunk_position,
                'progress': effective_position / effective_length if effective_length > 0 else 0,
            }
        return {}

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle state without file handles"""
        state = self.__dict__.copy()
        # Remove file handles that can't be pickled
        state.pop('current_bin_reader', None)
        state.pop('current_index', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and reinitialize file handles"""
        self.__dict__.update(state)
        # Save position before reloading (load_current_chunk resets position to effective_chunk_start)
        saved_position = getattr(self, 'current_chunk_position', 0)
        # Reload current chunk - worker_chunks already has correct shuffled order from checkpoint
        if hasattr(self, 'worker_chunks') and self.worker_chunks:
            self.load_current_chunk()
            # Restore saved position
            self.current_chunk_position = saved_position

    def __del__(self) -> None:
        """Clean up resources"""
        if hasattr(self, "current_bin_reader"):
            del self.current_bin_reader
        if hasattr(self, "current_index"):
            del self.current_index


# Keep the existing reader classes (unchanged from original)
class _RandomAccessBinReader:
    """
    Binary reader optimized for random access on very large files.
    """

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self.file_size = os.path.getsize(bin_path)
        self._init_mmap_random_access()
        self._set_random_access_hints()

    def _init_mmap_random_access(self):
        """Initialize memory mapping optimized for random access"""
        import mmap

        try:
            self.file_handle = open(self.bin_path, 'rb', buffering=0)
            self._mmap = mmap.mmap(
                self.file_handle.fileno(),
                0,  # Map entire file
                access=mmap.ACCESS_READ,
            )

            if hasattr(mmap, 'MADV_RANDOM'):
                try:
                    self._mmap.madvise(mmap.MADV_RANDOM)
                except (OSError, AttributeError):
                    pass

            logger.debug(f"Memory mapped {self.file_size / 1024 ** 3:.2f}GB file for random access")

        except Exception as e:
            logger.error(f"Failed to mmap {self.bin_path}: {e}")
            raise

    def _set_random_access_hints(self):
        """Set OS hints for random access patterns"""
        try:
            if hasattr(os, 'posix_fadvise'):
                fd = self.file_handle.fileno()
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Direct random access read with minimal overhead."""
        try:
            dtype_size = np.dtype(dtype).itemsize
            bytes_to_read = count * dtype_size
            data_bytes = self._mmap[offset:offset + bytes_to_read]

            if len(data_bytes) != bytes_to_read:
                end_position = offset + bytes_to_read
                shortfall = bytes_to_read - len(data_bytes)
                raise RuntimeError(
                    f"Could not read {bytes_to_read} bytes at offset {offset}\n"
                    f"  File: {self.bin_path}\n"
                    f"  File size: {self.file_size} bytes\n"
                    f"  Requested end position: {end_position} bytes\n"
                    f"  Shortfall: {shortfall} bytes\n"
                    f"  Dtype: {dtype.__name__}, Count: {count}, Dtype size: {dtype_size}\n"
                    f"  This indicates the chunk is corrupted or truncated."
                )

            return np.frombuffer(data_bytes, dtype=dtype, count=count)

        except RuntimeError:
            # Re-raise our detailed RuntimeError as-is
            raise
        except Exception as e:
            # Wrap other exceptions with additional context
            raise RuntimeError(
                f"Random access read failed\n"
                f"  File: {self.bin_path}\n"
                f"  Offset: {offset}\n"
                f"  Count: {count}\n"
                f"  Dtype: {dtype.__name__}\n"
                f"  Error: {e}"
            )

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


class _IndexReader:
    """Index reader for chunk files"""

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

        # Use memory mapping for index
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





def test_chunked_dataloader(chunks_dir: str, num_workers: int = 4, num_samples: int = 10):
    """Test function to verify the chunked dataloader works correctly"""
    print(f"Testing chunked dataloader with {num_workers} workers...")

    dataloaders = []
    for rank in range(num_workers):
        dl = ( ChunkedMMapDataset(
        chunks_dir=chunks_dir,
        dp_world_size=num_workers,
        dp_rank=rank,
        infinite=False,
    ))
        dataloaders.append(dl)
        print(f"Worker {rank}: {len(dl)} total samples, {len(dl.worker_chunks)} chunks")

    # Test reading a few samples from each worker
    for rank, dl in enumerate(dataloaders):
        print(f"\nWorker {rank} samples:")
        for i, batch in enumerate(dl):
            if i >= num_samples:
                break
            print(f"  Sample {i}: shape {batch['tokens'].shape}, "
                  f"first few tokens: {batch['tokens'][:10]}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True, help="Directory with chunk files")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to simulate")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples to read per worker")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    test_chunked_dataloader(args.chunks_dir, args.num_workers, args.num_samples)