#!/usr/bin/env python3
"""
Memory-efficient dataset chunker for TB+ scale datasets.

OPTIMIZED VERSION with:
- ProcessPoolExecutor for true parallelism
- Async flush manager with multiple writer processes
- Process-safe temp file management
- Improved error handling and performance
- Integrated validation and automatic cleanup

Uses streaming approach:
1. Multiple worker processes read datasets consecutively (fast sequential I/O)
2. Samples are randomly assigned to temporary chunks during reading
3. Async flush processes handle disk writes without blocking readers
4. Each temp chunk is then shuffled and written to final chunk
5. Output is validated
6. Temp chunks are automatically deleted if validation passes

"""

import logging
import os
import struct
import argparse
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict, deque
import queue
import time
import traceback
import sys

from titan_oellm.datasets.dataloader.mmap_dataset_chunked import DType, _IndexReader, _RandomAccessBinReader

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


@dataclass
class SequenceInfo:
    """Information about a sequence to be written"""
    data: np.ndarray
    length: int
    mode: Any


@dataclass
class FlushRequest:
    """Request to flush data to disk"""
    worker_id: int
    chunk_id: int
    sequences: List[SequenceInfo]
    dtype_str: str  # Store as string to avoid multiprocessing serialization issues


class AsyncFlushManager:
    """Manages async disk writes with multiple writer processes"""

    def __init__(self, temp_dir: Path, num_writers: int = 3, queue_size: int = 1000):
        self.temp_dir = temp_dir
        self.num_writers = num_writers

        # Use manager for shared state
        self.manager = mp.Manager()
        self.flush_queue = self.manager.Queue(maxsize=queue_size)
        self.error_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()

        self.writer_processes = []

    def start(self):
        """Start writer processes"""
        for i in range(self.num_writers):
            process = mp.Process(
                target=self._writer_worker,
                args=(i, self.flush_queue, self.error_queue, self.stop_event, self.temp_dir)
            )
            process.start()
            self.writer_processes.append(process)

        logger.info(f"Started {self.num_writers} async flush writer processes")

    def stop(self):
        """Stop all writer processes"""
        self.stop_event.set()

        # Send stop signals to all writers
        for _ in self.writer_processes:
            try:
                self.flush_queue.put(None, timeout=1.0)
            except:
                pass

        # Wait for processes to finish
        for process in self.writer_processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()

        logger.info("Stopped async flush writer processes")

    def submit_flush(self, flush_request: FlushRequest):
        """Submit flush request (non-blocking)"""
        try:
            self.flush_queue.put(flush_request, timeout=1.0)
        except queue.Full:
            raise RuntimeError("Flush queue is full - writers cannot keep up with readers")

    def check_errors(self):
        """Check for errors from writer processes"""
        try:
            error = self.error_queue.get_nowait()
            raise error
        except queue.Empty:
            pass

    def get_status(self):
        """Get status of flush manager"""
        alive_writers = sum(1 for p in self.writer_processes if p.is_alive())
        dead_writers = [i for i, p in enumerate(self.writer_processes) if not p.is_alive()]

        return {
            'queue_size': self.flush_queue.qsize() if hasattr(self.flush_queue, 'qsize') else -1,
            'alive_writers': alive_writers,
            'dead_writers': dead_writers,
            'total_writers': len(self.writer_processes)
        }

    @staticmethod
    def _writer_worker(worker_id: int, flush_queue, error_queue, stop_event, temp_dir: Path):
        """Writer worker process"""
        try:
            logger.info(f"Flush writer {worker_id} started")
            requests_processed = 0
            last_log_time = time.time()

            while not stop_event.is_set():
                try:
                    # Get flush request
                    flush_request = flush_queue.get(timeout=1.0)

                    if flush_request is None:  # Stop signal
                        break

                    # Process flush request with timing
                    write_start = time.time()
                    AsyncFlushManager._write_flush_request(flush_request, temp_dir)
                    write_duration = time.time() - write_start

                    # Warn if write is very slow
                    if write_duration > 5.0:
                        logger.warning(f"Flush writer {worker_id}: Slow write detected ({write_duration:.1f}s) "
                                     f"for worker {flush_request.worker_id} chunk {flush_request.chunk_id}")

                    requests_processed += 1

                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_log_time > 30:  # Log every 30 seconds
                        logger.info(f"Flush writer {worker_id}: Processed {requests_processed} flush requests")
                        last_log_time = current_time

                except queue.Empty:
                    continue
                except Exception as e:
                    # Capture full exception details
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
                    error_msg = f"Flush writer {worker_id} error:\n{tb_str}"
                    logger.error(error_msg)

                    error_info = {
                        'flush_writer_id': worker_id,
                        'exception_type': str(type(e).__name__),
                        'exception_message': str(e),
                        'traceback': tb_str
                    }
                    error_queue.put(error_info)
                    break

            logger.info(f"Flush writer {worker_id} stopped (processed {requests_processed} requests)")

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            error_msg = f"Flush writer {worker_id} failed:\n{tb_str}"
            logger.error(error_msg)

            error_info = {
                'flush_writer_id': worker_id,
                'exception_type': str(type(e).__name__),
                'exception_message': str(e),
                'traceback': tb_str
            }
            error_queue.put(error_info)

    @staticmethod
    def _write_flush_request(flush_request: FlushRequest, temp_dir: Path):
        """Write a flush request to disk"""
        worker_id = flush_request.worker_id
        chunk_id = flush_request.chunk_id
        sequences = flush_request.sequences
        dtype_str = flush_request.dtype_str

        if not sequences:
            return

        # Create worker/chunk directory
        worker_chunk_dir = temp_dir / f"worker_{worker_id:04d}" / f"chunk_{chunk_id:04d}"
        worker_chunk_dir.mkdir(parents=True, exist_ok=True)

        bin_path = worker_chunk_dir / "data.bin"
        metadata_path = worker_chunk_dir / "metadata.txt"
        dtype_path = worker_chunk_dir / "dtype.txt"

        # Write dtype info once
        if not dtype_path.exists():
            with open(dtype_path, 'w') as f:
                f.write(dtype_str)

        # Write binary data
        with open(bin_path, 'ab') as f:
            for seq_info in sequences:
                seq_info.data.tofile(f)

        # Write metadata
        with open(metadata_path, 'a') as f:
            for seq_info in sequences:
                f.write(f"{seq_info.length},{seq_info.mode}\n")


class ProcessSafeChunkBuffer:
    """Process-safe chunk buffer for individual workers"""

    def __init__(self, worker_id: int, buffer_size: int = 1000):
        self.worker_id = worker_id
        self.buffer_size = buffer_size
        self.buffers = {i: [] for i in range(4096)}  # Assume max chunks

    def add_sequence(self, chunk_id: int, sequence_info: SequenceInfo, dtype: np.dtype,
                     flush_manager: AsyncFlushManager):
        """Add sequence to buffer, flush if needed"""
        self.buffers[chunk_id].append(sequence_info)

        if len(self.buffers[chunk_id]) >= self.buffer_size:
            self._flush_buffer(chunk_id, dtype, flush_manager)

    def _flush_buffer(self, chunk_id: int, dtype: np.dtype, flush_manager: AsyncFlushManager):
        """Flush buffer to async manager"""
        if not self.buffers[chunk_id]:
            return

        # Convert dtype to string for safe multiprocessing
        if hasattr(dtype, 'str'):
            dtype_str = dtype.str
        else:
            dtype_str = str(np.dtype(dtype))

        flush_request = FlushRequest(
            worker_id=self.worker_id,
            chunk_id=chunk_id,
            sequences=list(self.buffers[chunk_id]),
            dtype_str=dtype_str
        )

        self.buffers[chunk_id].clear()
        flush_manager.submit_flush(flush_request)

    def flush_all(self, dtype: np.dtype, flush_manager: AsyncFlushManager):
        """Flush all remaining buffers"""
        for chunk_id in range(len(self.buffers)):
            self._flush_buffer(chunk_id, dtype, flush_manager)


class StreamingDatasetChunker:
    """Memory-efficient dataset chunker using streaming approach with multiprocessing"""

    def __init__(self,
                 input_paths: List[str] = None,
                 input_folder: str = None,
                 input_folders: List[str] = None,
                 output_dir: str = None,
                 num_chunks: int = 4096,
                 num_workers: int = None,
                 num_flush_writers: int = 3,
                 seed: int = 42,
                 temp_dir: str = None,
                 buffer_size: int = 1000,
                 progress_interval: int = 500000,
                 resume: bool = True,
                 force_reprocess: bool = False,
                 val_percent: float = 0.0):

        # Validate input specification (exactly one must be provided)
        provided_inputs = sum([
            input_paths is not None,
            input_folder is not None,
            input_folders is not None
        ])
        if provided_inputs == 0:
            raise ValueError("One of input_paths, input_folder, or input_folders must be provided")
        if provided_inputs > 1:
            raise ValueError("Cannot specify multiple input sources (use only one of input_paths, input_folder, or input_folders)")

        # Discover dataset paths
        if input_folders is not None:
            self.input_paths = self.discover_datasets_in_folders(input_folders)
        elif input_folder is not None:
            self.input_paths = self.discover_datasets_in_folder(input_folder)
        else:
            self.input_paths = input_paths

        self.output_dir = Path(output_dir)
        self.num_chunks = num_chunks
        self.seed = seed
        self.buffer_size = buffer_size
        self.progress_interval = progress_interval
        self.num_flush_writers = num_flush_writers
        self.resume = resume
        self.force_reprocess = force_reprocess
        self.val_fraction = val_percent / 100.0

        # Set number of workers
        if num_workers is None:
            self.num_workers = min(len(self.input_paths), os.cpu_count())
        else:
            self.num_workers = num_workers

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.val_fraction > 0:
            self.chunks_dir = self.output_dir / "chunks"
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.chunks_dir = self.output_dir

        # Setup temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            # Handle force_reprocess: clear temp directory if it exists
            if self.force_reprocess and self.temp_dir.exists():
                logger.info(f"Force reprocess enabled - clearing temp directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup_temp = False
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="chunker_tmp_"))
            self.cleanup_temp = True

        # Multiprocessing setup
        self.manager = mp.Manager()
        self.error_queue = self.manager.Queue()
        self.progress_queue = self.manager.Queue()

        logger.info(f"Initialized streaming chunker:")
        logger.info(f"  Datasets: {len(self.input_paths)}")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Flush writers: {self.num_flush_writers}")
        logger.info(f"  Target chunks: {num_chunks}")
        logger.info(f"  Buffer size: {buffer_size}")
        logger.info(f"  Temp dir: {self.temp_dir}")
        if self.val_fraction > 0:
            logger.info(f"  Validation split: {val_percent}%")

    def discover_datasets_in_folder(self, folder_path: str) -> List[str]:
        """Discover all idx/bin dataset pairs in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        logger.info(f"Discovering datasets in folder: {folder_path}")

        # Find all .idx files
        idx_files = list(folder.glob("*.idx"))
        dataset_prefixes = []

        for idx_file in idx_files:
            # Get the prefix (filename without .idx extension)
            prefix = str(idx_file).replace('.idx', '')
            bin_file = Path(f"{prefix}.bin")

            if bin_file.exists():
                dataset_prefixes.append(prefix)
                logger.debug(f"Found dataset pair: {prefix}")
            else:
                logger.warning(f"Found .idx file without matching .bin file: {idx_file}")

        if not dataset_prefixes:
            raise RuntimeError(f"No valid idx/bin dataset pairs found in {folder_path}")

        logger.info(f"Discovered {len(dataset_prefixes)} dataset pairs in {folder_path}")
        return sorted(dataset_prefixes)  # Sort for consistent ordering

    def discover_datasets_in_folders(self, folder_paths: List[str]) -> List[str]:
        """Discover all idx/bin dataset pairs across multiple folders"""
        all_dataset_prefixes = []

        logger.info(f"Discovering datasets across {len(folder_paths)} folders")

        for folder_path in folder_paths:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            # Find all .idx files in this folder
            idx_files = list(folder.glob("*.idx"))
            folder_datasets = 0

            for idx_file in idx_files:
                # Get the prefix (filename without .idx extension)
                prefix = str(idx_file).replace('.idx', '')
                bin_file = Path(f"{prefix}.bin")

                if bin_file.exists():
                    all_dataset_prefixes.append(prefix)
                    folder_datasets += 1
                    logger.debug(f"Found dataset pair: {prefix}")
                else:
                    logger.warning(f"Found .idx file without matching .bin file: {idx_file}")

            logger.info(f"  {folder_path}: {folder_datasets} dataset pairs")

        if not all_dataset_prefixes:
            raise RuntimeError(f"No valid idx/bin dataset pairs found in any of the {len(folder_paths)} folders")

        logger.info(f"Total: {len(all_dataset_prefixes)} dataset pairs across {len(folder_paths)} folders")
        return sorted(all_dataset_prefixes)  # Sort for consistent ordering

    def validate_dataset_pair(self, path_prefix: str) -> Dict[str, Any]:
        """Validate that an idx/bin file pair is consistent and complete.

        Performs comprehensive validation:
        1. File existence check
        2. Index file header validation (magic bytes)
        3. Index file structure validation (expected size based on sequence count)
        4. Binary file size validation (matches index pointers)
        5. Corruption detection (pointer consistency, size bounds)
        """
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"

        logger.debug(f"Validating dataset pair: {path_prefix}")

        # Check files exist
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Index file not found: {idx_path}")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        # Get file sizes
        idx_size = os.path.getsize(idx_path)
        bin_size = os.path.getsize(bin_path)

        # Validate index file header and structure
        try:
            with open(idx_path, 'rb') as f:
                # Check header magic bytes
                header = f.read(9)
                if header != _INDEX_HEADER:
                    raise RuntimeError(
                        f"Invalid index file header in {idx_path}. "
                        f"Expected {_INDEX_HEADER!r}, got {header!r}. "
                        f"File may be corrupted or not a valid mmap index."
                    )

                # Read metadata
                version = struct.unpack("<Q", f.read(8))[0]
                dtype_code = struct.unpack("<B", f.read(1))[0]
                sequence_count = struct.unpack("<Q", f.read(8))[0]
                document_count = struct.unpack("<Q", f.read(8))[0]

                # Validate version
                if version != 1:
                    logger.warning(f"Unknown index version {version} in {idx_path}, expected 1")

                # Calculate expected index file size
                # Header (9) + version (8) + dtype (1) + seq_count (8) + doc_count (8) +
                # lengths (seq_count * 4) + pointers (seq_count * 8) + doc_indices (doc_count * 8)
                expected_idx_size = 9 + 8 + 1 + 8 + 8 + (sequence_count * 4) + (sequence_count * 8) + (document_count * 8)

                if idx_size < expected_idx_size:
                    raise RuntimeError(
                        f"Index file {idx_path} is truncated. "
                        f"Expected at least {expected_idx_size} bytes based on {sequence_count} sequences, "
                        f"got {idx_size} bytes. File may be corrupted."
                    )

                if idx_size > expected_idx_size + 1024:  # Allow small slack for alignment
                    logger.warning(
                        f"Index file {idx_path} is larger than expected. "
                        f"Expected ~{expected_idx_size} bytes, got {idx_size} bytes."
                    )

        except struct.error as e:
            raise RuntimeError(
                f"Failed to parse index file structure in {idx_path}: {e}. "
                f"File may be truncated or corrupted."
            )

        # Load index to get full metadata and validate pointers
        try:
            index = _IndexReader(idx_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read index file {idx_path}: {e}")

        dtype_size = np.dtype(index.dtype).itemsize

        # Calculate expected binary file size from index
        if len(index) == 0:
            expected_bin_size = 0
        else:
            # Get the last sequence info
            last_pointer, last_length, _ = index[len(index) - 1]
            expected_bin_size = last_pointer + (last_length * dtype_size)

            # Validate pointer consistency (spot check first and last few sequences)
            sequences_to_check = min(10, len(index))
            prev_end = 0
            for i in range(sequences_to_check):
                pointer, length, _ = index[i]
                if pointer < prev_end:
                    raise RuntimeError(
                        f"Index file {idx_path} has overlapping pointers at sequence {i}. "
                        f"Pointer {pointer} < previous end {prev_end}. File is corrupted."
                    )
                if pointer > bin_size:
                    raise RuntimeError(
                        f"Index file {idx_path} has pointer beyond bin file at sequence {i}. "
                        f"Pointer {pointer} > bin size {bin_size}. File is corrupted."
                    )
                if length < 0:
                    raise RuntimeError(
                        f"Index file {idx_path} has negative length at sequence {i}. "
                        f"Length {length}. File is corrupted."
                    )
                prev_end = pointer + (length * dtype_size)

            # Also check last few sequences
            if len(index) > sequences_to_check:
                for i in range(max(sequences_to_check, len(index) - 5), len(index)):
                    pointer, length, _ = index[i]
                    if pointer > bin_size:
                        raise RuntimeError(
                            f"Index file {idx_path} has pointer beyond bin file at sequence {i}. "
                            f"Pointer {pointer} > bin size {bin_size}. File is corrupted."
                        )
                    if length < 0:
                        raise RuntimeError(
                            f"Index file {idx_path} has negative length at sequence {i}. "
                            f"Length {length}. File is corrupted."
                        )

        # Validate binary file size
        if bin_size < expected_bin_size:
            raise RuntimeError(
                f"Binary file {bin_path} is too small. "
                f"Expected at least {expected_bin_size} bytes based on index, got {bin_size} bytes. "
                f"File appears to be truncated or corrupted."
            )

        if bin_size > expected_bin_size * 1.01 + 4096:  # Allow 1% + 4KB slack
            logger.warning(
                f"Binary file {bin_path} is larger than expected. "
                f"Expected ~{expected_bin_size} bytes, got {bin_size} bytes. "
                f"May contain extra data or padding."
            )

        validation_info = {
            'path_prefix': path_prefix,
            'sequence_count': len(index),
            'document_count': index.document_count,
            'dtype': index.dtype,
            'idx_size_bytes': idx_size,
            'bin_size_bytes': bin_size,
            'expected_bin_size_bytes': expected_bin_size,
            'valid': True
        }

        logger.debug(f"Validation successful for {path_prefix}: "
                     f"{validation_info['sequence_count']} sequences, "
                     f"{validation_info['bin_size_bytes'] / 1024 ** 3:.2f} GB")

        # Clean up
        del index

        return validation_info

    def validate_dtype_consistency(self, validation_results: List[Dict[str, Any]]) -> np.dtype:
        """Validate that all datasets have the same dtype"""
        dtypes = [result['dtype'] for result in validation_results]
        unique_dtypes = set(str(dtype) for dtype in dtypes)

        if len(unique_dtypes) > 1:
            raise ValueError(f"Inconsistent dtypes across datasets: {unique_dtypes}")

        return dtypes[0]

    def _validate_temp_chunk(self, worker_id: int, chunk_id: int, chunk_dir: Path) -> bool:
        """
        Validate if a temp chunk is complete and usable.

        Args:
            worker_id: Worker ID
            chunk_id: Chunk ID
            chunk_dir: Path to the chunk directory

        Returns:
            True if chunk is valid and complete, False otherwise
        """
        bin_path = chunk_dir / "data.bin"
        metadata_path = chunk_dir / "metadata.txt"
        dtype_path = chunk_dir / "dtype.txt"

        # Check existence
        if not all(p.exists() for p in [bin_path, metadata_path, dtype_path]):
            logger.debug(f"Worker {worker_id} chunk {chunk_id}: Missing files")
            return False

        try:
            # Read dtype
            with open(dtype_path, 'r') as f:
                dtype_str = f.read().strip()
                try:
                    dtype = np.dtype(dtype_str)
                except (TypeError, ValueError):
                    # Fallback for old format
                    if 'numpy.' in dtype_str:
                        import re
                        match = re.search(r'numpy\.(\w+)', dtype_str)
                        if match:
                            dtype = np.dtype(match.group(1))
                        else:
                            logger.warning(f"Worker {worker_id} chunk {chunk_id}: Cannot parse dtype: {dtype_str}")
                            return False
                    else:
                        dtype = np.dtype(dtype_str)

            # Count metadata lines and calculate expected size
            expected_bytes = 0
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        length = int(parts[0])
                        expected_bytes += length * dtype.itemsize

            # Check bin size
            actual_bytes = bin_path.stat().st_size

            if actual_bytes != expected_bytes:
                logger.warning(f"Worker {worker_id} chunk {chunk_id}: Size mismatch "
                              f"(expected {expected_bytes}, got {actual_bytes})")
                return False

            return True

        except Exception as e:
            logger.warning(f"Worker {worker_id} chunk {chunk_id}: Validation error: {e}")
            return False

    def _get_completed_workers(self) -> set:
        """
        Determine which workers have completed processing based on temp directory state.

        A worker is considered complete if:
        Method 1 (with COMPLETE marker):
            - COMPLETE marker file exists
            - All temp chunks for that worker pass validation

        Method 2 (without marker - backward compatible):
            - All temp chunks pass validation
            - Total sequences in temp chunks equals expected sequences in source dataset

        Also validates that temp directory num_chunks matches current configuration.

        Returns:
            Set of worker IDs that are fully complete
        """
        if not self.temp_dir.exists():
            logger.debug("Temp directory does not exist - no workers to resume")
            return set()

        completed = set()

        # First, infer num_chunks from temp directory structure
        max_chunk_id = -1
        for worker_dir in self.temp_dir.glob("worker_*"):
            for chunk_dir in worker_dir.glob("chunk_*"):
                chunk_id = int(chunk_dir.name.split("_")[1])
                max_chunk_id = max(max_chunk_id, chunk_id)

        # Validate num_chunks configuration if we found any chunks
        if max_chunk_id >= 0:
            temp_num_chunks = max_chunk_id + 1
            if temp_num_chunks > self.num_chunks:
                raise ValueError(
                    f"Temp directory has chunks up to {max_chunk_id} "
                    f"but current num_chunks is {self.num_chunks}. "
                    f"Cannot resume - use --force-reprocess to start fresh."
                )
            elif temp_num_chunks < self.num_chunks:
                logger.warning(
                    f"Temp directory has chunks up to {max_chunk_id} "
                    f"but current num_chunks is {self.num_chunks}. "
                    f"This is acceptable but may indicate incomplete previous run."
                )

        # Scan for completed workers
        # Build mapping of worker_id to dataset_path
        worker_dataset_map = {i: path for i, path in enumerate(self.input_paths)}

        for worker_dir in sorted(self.temp_dir.glob("worker_*")):
            worker_id = int(worker_dir.name.split("_")[1])

            # Skip if worker_id is out of range
            if worker_id >= len(self.input_paths):
                logger.warning(f"Worker {worker_id}: Directory exists but no corresponding dataset - skipping")
                continue

            dataset_path = worker_dataset_map[worker_id]
            marker_file = worker_dir / "COMPLETE"

            # Method 1: Check for COMPLETE marker (fast path for new temp files)
            if marker_file.exists():
                logger.debug(f"Worker {worker_id}: Found COMPLETE marker - validating chunks")

                # Validate all chunks for this worker
                all_valid = True
                chunk_count = 0
                for chunk_dir in worker_dir.glob("chunk_*"):
                    chunk_id = int(chunk_dir.name.split("_")[1])
                    if not self._validate_temp_chunk(worker_id, chunk_id, chunk_dir):
                        all_valid = False
                        logger.warning(f"Worker {worker_id} chunk {chunk_id} invalid - will reprocess worker")
                        break
                    chunk_count += 1

                if all_valid and chunk_count > 0:
                    completed.add(worker_id)
                    logger.info(f"Worker {worker_id}: Previously completed ({chunk_count} chunks) - will skip")
                elif chunk_count == 0:
                    logger.warning(f"Worker {worker_id}: COMPLETE marker but no chunks found - will reprocess")

                continue

            # Method 2: Validate by counting sequences (backward compatible, no marker needed)
            logger.debug(f"Worker {worker_id}: No COMPLETE marker - validating by sequence count")

            try:
                # Validate all chunks and count sequences
                all_valid = True
                total_sequences_in_chunks = 0
                chunk_count = 0

                for chunk_dir in worker_dir.glob("chunk_*"):
                    chunk_id = int(chunk_dir.name.split("_")[1])

                    # Validate chunk integrity
                    if not self._validate_temp_chunk(worker_id, chunk_id, chunk_dir):
                        all_valid = False
                        logger.warning(f"Worker {worker_id} chunk {chunk_id} invalid - will reprocess worker")
                        break

                    # Count sequences in this chunk
                    metadata_path = chunk_dir / "metadata.txt"
                    with open(metadata_path, 'r') as f:
                        sequences_count = sum(1 for line in f if line.strip())
                    total_sequences_in_chunks += sequences_count
                    chunk_count += 1

                if not all_valid or chunk_count == 0:
                    logger.debug(f"Worker {worker_id}: Invalid or empty chunks - will process")
                    continue

                # Get expected sequence count from source dataset
                idx_path = f"{dataset_path}.idx"
                index = _IndexReader(idx_path)
                expected_sequences = len(index)
                del index

                # Compare: if matches, worker is complete
                if total_sequences_in_chunks == expected_sequences:
                    completed.add(worker_id)
                    logger.info(f"Worker {worker_id}: Complete ({total_sequences_in_chunks} sequences, "
                               f"{chunk_count} chunks) - will skip")
                else:
                    logger.warning(f"Worker {worker_id}: Incomplete "
                                  f"({total_sequences_in_chunks}/{expected_sequences} sequences, "
                                  f"{chunk_count} chunks) - will reprocess")

            except Exception as e:
                logger.warning(f"Worker {worker_id}: Error validating completion state: {e}")
                logger.debug(f"Worker {worker_id}: Will reprocess due to validation error")

        return completed

    def _validate_output_chunk(self, chunk_id: int, dtype: np.dtype) -> bool:
        """
        Validate if an output chunk is complete and valid.

        Args:
            chunk_id: Chunk ID
            dtype: Expected data type

        Returns:
            True if chunk is valid and complete, False otherwise
        """
        chunk_prefix = self.chunks_dir / f"chunk_{chunk_id:04d}"
        idx_path = f"{chunk_prefix}.idx"
        bin_path = f"{chunk_prefix}.bin"

        # Check existence
        if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
            logger.debug(f"Chunk {chunk_id}: Missing files")
            return False

        try:
            # Read index header and metadata
            with open(idx_path, 'rb') as f:
                # Validate header
                header = f.read(9)
                if header != _INDEX_HEADER:
                    logger.warning(f"Chunk {chunk_id}: Invalid header")
                    return False

                # Read metadata
                version = struct.unpack("<Q", f.read(8))[0]
                dtype_code = struct.unpack("<B", f.read(1))[0]
                sequence_count = struct.unpack("<Q", f.read(8))[0]
                document_count = struct.unpack("<Q", f.read(8))[0]

                # Read sequence lengths
                lengths = np.fromfile(f, dtype=np.int32, count=sequence_count)

            # Calculate expected bin size
            expected_bytes = int(np.sum(lengths)) * dtype.itemsize

            # Check actual bin size
            actual_bytes = os.path.getsize(bin_path)

            if actual_bytes != expected_bytes:
                logger.warning(f"Chunk {chunk_id}: Size mismatch "
                              f"(expected {expected_bytes}, got {actual_bytes})")
                return False

            logger.debug(f"Chunk {chunk_id}: Valid ({sequence_count} sequences, {actual_bytes} bytes)")
            return True

        except Exception as e:
            logger.warning(f"Chunk {chunk_id}: Validation error: {e}")
            return False

    def _scan_output_chunks(self, dtype: np.dtype) -> Dict[int, bool]:
        """
        Scan output directory to find valid completed chunks.

        Args:
            dtype: Expected data type

        Returns:
            Dict mapping chunk_id to is_complete
        """
        output_state = {}

        for chunk_id in range(self.num_chunks):
            is_complete = self._validate_output_chunk(chunk_id, dtype)
            output_state[chunk_id] = is_complete

            if is_complete:
                logger.debug(f"Chunk {chunk_id}: Already complete - will skip")

        num_complete = sum(1 for v in output_state.values() if v)
        if num_complete > 0:
            logger.info(f"Found {num_complete} existing valid output chunks")

        return output_state

    @staticmethod
    def distribute_dataset_to_temp_chunks(dataset_path: str,
                                          worker_id: int,
                                          num_chunks: int,
                                          seed: int,
                                          buffer_size: int,
                                          progress_interval: int,
                                          temp_dir: Path,
                                          flush_queue,
                                          error_queue,
                                          progress_queue,
                                          val_fraction: float = 0.0) -> Dict[str, Any]:
        """
        Worker function: Read dataset consecutively and distribute to random temp chunks
        """
        try:
            # Setup flush manager communication
            class WorkerFlushManager:
                def __init__(self, flush_queue, worker_id):
                    self.flush_queue = flush_queue
                    self.worker_id = worker_id

                def submit_flush(self, flush_request):
                    retries = 0
                    max_retries = 10
                    while retries < max_retries:
                        try:
                            self.flush_queue.put(flush_request, timeout=5.0)
                            return
                        except queue.Full:
                            retries += 1
                            logger.warning(
                                f"Worker {self.worker_id}: Flush queue full (retry {retries}/{max_retries}). "
                                f"Flush writers may be struggling to keep up."
                            )
                            time.sleep(1.0)

                    raise RuntimeError(
                        f"Worker {self.worker_id}: Flush queue full after {max_retries} retries. "
                        f"Flush writers cannot keep up with data production. "
                        f"Try increasing --num-flush-writers or --buffer-size."
                    )

            flush_manager = WorkerFlushManager(flush_queue, worker_id)

            # Setup worker-specific random state
            worker_rng = np.random.RandomState(seed + worker_id)

            # Load dataset
            idx_path = f"{dataset_path}.idx"
            bin_path = f"{dataset_path}.bin"

            index = _IndexReader(idx_path)
            bin_reader = _RandomAccessBinReader(bin_path)

            total_sequences = len(index)
            dtype = index.dtype
            dtype_size = np.dtype(dtype).itemsize

            logger.info(f"Worker {worker_id}: Processing {dataset_path} "
                        f"({total_sequences} sequences, dtype: {dtype})")

            # Create buffer manager
            buffer_manager = ProcessSafeChunkBuffer(worker_id, buffer_size)

            # Track stats
            sequences_written = 0
            bytes_written = 0
            chunk_counts = defaultdict(int)
            last_log_time = time.time()
            last_progress_time = time.time()

            # Process sequences consecutively
            for seq_idx in range(total_sequences):
                try:
                    # Read sequence
                    sequence_pointer, sequence_length, sequence_mode = index[seq_idx]
                    sequence_data = bin_reader.read(
                        dtype=dtype,
                        count=sequence_length,
                        offset=sequence_pointer
                    )

                    # Randomly select target temp chunk (or validation set)
                    if val_fraction > 0 and worker_rng.random() < val_fraction:
                        target_chunk_id = num_chunks  # validation "chunk"
                    else:
                        target_chunk_id = worker_rng.randint(0, num_chunks)

                    # Create sequence info
                    seq_info = SequenceInfo(
                        data=sequence_data,
                        length=sequence_length,
                        mode=sequence_mode,
                    )

                    # Add to buffer manager
                    buffer_manager.add_sequence(target_chunk_id, seq_info, dtype, flush_manager)

                    # Update stats
                    sequences_written += 1
                    bytes_written += sequence_length * dtype_size
                    chunk_counts[target_chunk_id] += 1

                    # Update progress less frequently
                    current_time = time.time()
                    if seq_idx % progress_interval == 0 or current_time - last_progress_time > 30:
                        progress_queue.put(('progress', worker_id, seq_idx, total_sequences))
                        last_progress_time = current_time

                    # Periodic diagnostic logging
                    if current_time - last_log_time > 30:  # Log every 30 seconds
                        elapsed = current_time - last_log_time
                        rate = progress_interval / elapsed if elapsed > 0 else 0
                        logger.info(f"Worker {worker_id}: Progress {seq_idx}/{total_sequences} "
                                   f"({100*seq_idx/total_sequences:.1f}%), "
                                   f"{bytes_written / 1024**3:.2f} GB processed, "
                                   f"{rate:.0f} seq/s")
                        last_log_time = current_time

                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error at sequence {seq_idx}/{total_sequences}: {e}")
                    raise

            # Final progress update
            progress_queue.put(('progress', worker_id, total_sequences, total_sequences))

            # Flush all remaining buffers
            buffer_manager.flush_all(dtype, flush_manager)

            # Write COMPLETE marker to indicate this worker finished successfully
            worker_dir = temp_dir / f"worker_{worker_id:04d}"
            marker_file = worker_dir / "COMPLETE"
            try:
                with open(marker_file, 'w') as f:
                    f.write(f"Completed at: {time.time()}\n")
                    f.write(f"Dataset: {dataset_path}\n")
                    f.write(f"Sequences: {sequences_written}\n")
                logger.debug(f"Worker {worker_id}: Wrote COMPLETE marker")
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Failed to write COMPLETE marker: {e}")

            # Cleanup
            del bin_reader
            del index

            result = {
                'worker_id': worker_id,
                'dataset_path': dataset_path,
                'sequences_written': sequences_written,
                'bytes_written': bytes_written,
                'chunk_distribution': dict(chunk_counts),
                'dtype': str(dtype)
            }

            logger.info(f"Worker {worker_id}: Completed {dataset_path} - "
                        f"{sequences_written} sequences, "
                        f"{bytes_written / 1024 ** 3:.2f} GB")

            return result

        except Exception as e:
            # Capture full exception details
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            error_msg = f"Worker {worker_id} failed on {dataset_path}:\n{tb_str}"
            logger.error(error_msg)

            # Put serializable error info in queue
            error_info = {
                'worker_id': worker_id,
                'dataset_path': dataset_path,
                'exception_type': str(type(e).__name__),
                'exception_message': str(e),
                'traceback': tb_str
            }
            error_queue.put(error_info)
            raise

    @staticmethod
    def shuffle_and_finalize_chunk(chunk_id: int,
                                   temp_dir: Path,
                                   output_dir: Path,
                                   seed: int,
                                   num_workers: int,
                                   output_prefix: str = None) -> Dict[str, Any]:
        """
        Shuffle a temporary chunk and write final chunk (combines all worker outputs)
        """
        # Collect all worker temp chunks for this chunk_id
        all_sequences_info = []
        dtype = None

        for worker_id in range(num_workers):
            worker_chunk_path = temp_dir / f"worker_{worker_id:04d}" / f"chunk_{chunk_id:04d}"

            if not worker_chunk_path.exists():
                continue

            bin_path = worker_chunk_path / "data.bin"
            metadata_path = worker_chunk_path / "metadata.txt"
            dtype_path = worker_chunk_path / "dtype.txt"

            if not all(p.exists() for p in [bin_path, metadata_path, dtype_path]):
                continue

            # Read and validate dtype
            with open(dtype_path, 'r') as f:
                dtype_str = f.read().strip()
                try:
                    worker_dtype = np.dtype(dtype_str)
                except (TypeError, ValueError):
                    # Fallback for old format or unexpected dtype strings
                    if 'numpy.' in dtype_str:
                        # Handle "<class 'numpy.uint16'>" format
                        import re
                        match = re.search(r'numpy\.(\w+)', dtype_str)
                        if match:
                            worker_dtype = np.dtype(match.group(1))
                        else:
                            raise ValueError(f"Cannot parse dtype: {dtype_str}")
                    else:
                        # Try direct type name
                        worker_dtype = np.dtype(dtype_str)

                if dtype is None:
                    dtype = worker_dtype
                elif dtype != worker_dtype:
                    raise ValueError(f"Chunk {chunk_id}: dtype mismatch between workers")

            # Read metadata to get sequence boundaries
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        length = int(parts[0])
                        mode = None if parts[1] == 'None' else parts[1]
                        all_sequences_info.append((length, mode, worker_id, bin_path))

        if not all_sequences_info:
            logger.debug(f"Chunk {chunk_id}: Empty, skipping")
            return {'chunk_id': chunk_id, 'sequences': 0}

        # Create sequence index for reading from each worker file
        sequence_data_list = []

        # Read all sequences from all workers
        worker_file_handles = {}
        worker_offsets = {}

        try:
            for length, mode, worker_id, bin_path in all_sequences_info:
                # Open worker file if not already open
                if worker_id not in worker_file_handles:
                    worker_file_handles[worker_id] = open(bin_path, 'rb')
                    worker_offsets[worker_id] = 0

                # Read sequence data
                f = worker_file_handles[worker_id]
                f.seek(worker_offsets[worker_id])
                sequence_bytes = f.read(length * dtype.itemsize)
                sequence_data = np.frombuffer(sequence_bytes, dtype=dtype)

                sequence_data_list.append((sequence_data, length, mode))
                worker_offsets[worker_id] += length * dtype.itemsize

        finally:
            # Close all file handles
            for f in worker_file_handles.values():
                f.close()

        # Shuffle sequence order (IMPORTANT: maintain randomization)
        chunk_rng = np.random.RandomState(seed + chunk_id)
        shuffled_indices = list(range(len(sequence_data_list)))
        chunk_rng.shuffle(shuffled_indices)

        # Write final chunk in shuffled order
        if output_prefix is not None:
            chunk_prefix = output_dir / output_prefix
        else:
            chunk_prefix = output_dir / f"chunk_{chunk_id:04d}"
        idx_path = f"{chunk_prefix}.idx"
        bin_path = f"{chunk_prefix}.bin"

        sequence_lengths = []
        sequence_pointers = []
        current_pointer = 0

        with open(bin_path, 'wb') as final_f:
            for shuffled_idx in shuffled_indices:
                sequence_data, length, mode = sequence_data_list[shuffled_idx]

                # Write to final file
                sequence_data.tofile(final_f)

                # Record metadata
                sequence_lengths.append(length)
                sequence_pointers.append(current_pointer)
                current_pointer += length * dtype.itemsize

        # Write index file
        sequence_count = len(sequence_lengths)

        with open(idx_path, 'wb') as idx_file:
            # Write header
            idx_file.write(_INDEX_HEADER)

            # Write version
            idx_file.write(struct.pack("<Q", 1))

            # Write dtype code - FIXED BUG
            dtype_code = DType.code_from_dtype(dtype.type)
            idx_file.write(struct.pack("<B", dtype_code))

            # Write counts
            idx_file.write(struct.pack("<Q", sequence_count))
            idx_file.write(struct.pack("<Q", sequence_count))  # document_count = sequence_count

            # Write sequence lengths
            np.array(sequence_lengths, dtype=np.int32).tofile(idx_file)

            # Write sequence pointers
            np.array(sequence_pointers, dtype=np.int64).tofile(idx_file)

            # Write document indices
            np.arange(sequence_count, dtype=np.int64).tofile(idx_file)


        logger.debug(f"Chunk {chunk_id}: Finalized {len(sequence_data_list)} sequences")

        return {
            'chunk_id': chunk_id,
            'sequences': len(sequence_data_list),
        }


    def _monitor_progress(self, progress_queue, total_workers: int):
        """Monitor and display progress from all workers"""
        worker_progress = {}
        completed_workers = 0
        progress_bar = None

        while completed_workers < total_workers:
            try:
                msg = progress_queue.get(timeout=1.0)
                if msg[0] == 'progress':
                    _, worker_id, current, total = msg
                    worker_progress[worker_id] = (current, total)

                    if current == total:
                        completed_workers += 1

                elif msg[0] == 'complete':
                    completed_workers += 1

            except queue.Empty:
                continue

            # Update progress display
            if len(worker_progress) > 0:
                total_progress = sum(current for current, total in worker_progress.values())
                total_seq = sum(total for current, total in worker_progress.values())

                # Initialize progress bar if we have total sequences
                if progress_bar is None and total_seq > 0:
                    progress_bar = tqdm(total=total_seq, desc="Processing", unit="seq", unit_scale=True)

                # Update progress bar
                if progress_bar is not None:
                    progress_bar.n = total_progress
                    progress_bar.refresh()

        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()

    def _validate_output(self, validation_results: List[Dict],
                        finalization_results: List[Dict],
                        val_result: Optional[Dict] = None) -> bool:
        """Validate the output chunks and return True if validation passes"""
        logger.info("=== VALIDATING OUTPUT ===")

        # Check metadata file exists
        metadata_file = self.output_dir / "chunking_metadata.txt"
        if not metadata_file.exists():
            logger.error("Metadata file not found!")
            return False
        logger.info("✓ Metadata file exists")

        # Verify sequence counts (chunks + validation must equal original)
        original_sequences = sum(v['sequence_count'] for v in validation_results)
        chunk_sequences = sum(r['sequences'] for r in finalization_results)
        val_sequences = val_result['sequences'] if val_result else 0
        final_sequences = chunk_sequences + val_sequences

        logger.info(f"Original sequences:   {original_sequences:,}")
        logger.info(f"Training sequences:   {chunk_sequences:,}")
        if val_result:
            logger.info(f"Validation sequences: {val_sequences:,}")
        logger.info(f"Total sequences:      {final_sequences:,}")

        if original_sequences != final_sequences:
            logger.error(f"Sequence count mismatch! Difference: {abs(original_sequences - final_sequences):,}")
            return False
        logger.info("✓ Sequence counts match")

        # Count chunk files
        idx_files = list(self.chunks_dir.glob("chunk_*.idx"))
        bin_files = list(self.chunks_dir.glob("chunk_*.bin"))

        logger.info(f"Found {len(idx_files)} .idx files")
        logger.info(f"Found {len(bin_files)} .bin files")

        if len(idx_files) != len(bin_files):
            logger.error("Mismatch between idx and bin file counts")
            return False
        logger.info("✓ Chunk file counts consistent")

        # Validate ALL chunk files
        logger.info("Validating ALL chunk file structures...")
        chunks_to_validate = sorted(idx_files)

        validation_errors = []
        for idx_file in chunks_to_validate:
            try:
                bin_file = Path(str(idx_file).replace('.idx', '.bin'))

                # Check idx file structure
                with open(idx_file, 'rb') as f:
                    header = f.read(9)
                    if header != _INDEX_HEADER:
                        error_msg = f"Invalid header in {idx_file.name}"
                        logger.error(error_msg)
                        validation_errors.append(error_msg)
                        continue

                    version = struct.unpack("<Q", f.read(8))[0]
                    dtype_code = struct.unpack("<B", f.read(1))[0]
                    sequence_count = struct.unpack("<Q", f.read(8))[0]

                # Check that bin file size is reasonable
                bin_size = bin_file.stat().st_size
                if bin_size == 0 and sequence_count > 0:
                    error_msg = f"Empty bin file but idx claims {sequence_count} sequences: {idx_file.name}"
                    logger.error(error_msg)
                    validation_errors.append(error_msg)
                    continue

                logger.debug(f"✓ {idx_file.name}: {sequence_count:,} sequences, {bin_size / 1024**2:.1f} MB")

            except Exception as e:
                error_msg = f"Error validating {idx_file.name}: {e}"
                logger.error(error_msg)
                validation_errors.append(error_msg)

        # Validate validation set files if present
        if val_result and val_result['sequences'] > 0:
            val_idx = self.output_dir / "validation.idx"
            val_bin = self.output_dir / "validation.bin"
            if not val_idx.exists() or not val_bin.exists():
                validation_errors.append("Validation set files (validation.idx/bin) missing")
            else:
                logger.info(f"✓ Validation set: {val_sequences:,} sequences, "
                           f"{val_bin.stat().st_size / 1024**3:.4f} GB")

        if validation_errors:
            logger.error(f"Found {len(validation_errors)} validation errors")
            for err in validation_errors[:10]:  # Show first 10 errors
                logger.error(f"  - {err}")
            if len(validation_errors) > 10:
                logger.error(f"  ... and {len(validation_errors) - 10} more errors")
            return False

        logger.info(f"✓ Validated {len(chunks_to_validate)} chunk files successfully")

        # Count non-empty chunks
        non_empty = sum(1 for r in finalization_results if r['sequences'] > 0)
        logger.info(f"✓ Non-empty chunks: {non_empty}/{self.num_chunks}")

        # Calculate total size
        chunks_size = sum(f.stat().st_size for f in bin_files) / (1024**3)
        logger.info(f"✓ Training chunks size: {chunks_size:.2f} GB")
        if val_result and val_result['sequences'] > 0:
            val_size = (self.output_dir / "validation.bin").stat().st_size / (1024**3)
            logger.info(f"✓ Validation set size: {val_size:.4f} GB")
            logger.info(f"✓ Total output size: {chunks_size + val_size:.2f} GB")
        else:
            logger.info(f"✓ Total output size: {chunks_size:.2f} GB")

        logger.info("=== VALIDATION PASSED ===")
        return True

    def process(self) -> None:
        """Main processing pipeline with multiprocessing"""
        flush_manager = None
        validation_passed = False

        try:
            # Step 1: Validate all datasets
            logger.info("=== STEP 1: VALIDATING DATASETS ===")
            validation_results = []
            for dataset_path in self.input_paths:
                validation_info = self.validate_dataset_pair(dataset_path)
                validation_results.append(validation_info)

            # Validate dtype consistency early
            common_dtype = self.validate_dtype_consistency(validation_results)
            logger.info(f"Validated dtype consistency: {common_dtype}")

            # Log validation summary
            total_sequences = sum(v['sequence_count'] for v in validation_results)
            total_size_gb = sum(v['bin_size_bytes'] for v in validation_results) / 1024 ** 3
            logger.info(f"Validation complete: {total_sequences} total sequences, {total_size_gb:.2f} GB")

            # Step 2: Start async flush manager
            logger.info("=== STEP 2: STARTING ASYNC FLUSH MANAGER ===")
            flush_manager = AsyncFlushManager(self.temp_dir, self.num_flush_writers)
            flush_manager.start()

            # Step 3: Distribute datasets to temporary chunks using processes
            logger.info("=== STEP 3: DISTRIBUTING TO TEMP CHUNKS ===")

            # Check for resumable state
            datasets_to_process = []
            if self.resume and not self.force_reprocess:
                try:
                    completed_workers = self._get_completed_workers()
                    if completed_workers:
                        logger.info(f"=== RESUMING FROM PREVIOUS RUN ===")
                        logger.info(f"Found {len(completed_workers)} completed workers: {sorted(completed_workers)}")
                        # Build list of datasets to process (skip completed workers)
                        for worker_id, dataset_path in enumerate(self.input_paths):
                            if worker_id not in completed_workers:
                                datasets_to_process.append((worker_id, dataset_path))
                        logger.info(f"Skipping {len(completed_workers)} completed workers")
                        logger.info(f"Processing {len(datasets_to_process)} remaining workers")
                    else:
                        # No completed workers found, process all
                        datasets_to_process = list(enumerate(self.input_paths))
                        logger.info("No completed workers found - processing all datasets")
                except Exception as e:
                    logger.warning(f"Failed to detect resumable state: {e}")
                    logger.info("Starting fresh - processing all datasets")
                    datasets_to_process = list(enumerate(self.input_paths))
            else:
                # Resume disabled or force_reprocess enabled
                datasets_to_process = list(enumerate(self.input_paths))
                if self.force_reprocess:
                    logger.info("Force reprocess enabled - processing all datasets")

            # Use ProcessPoolExecutor for true parallelism
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all dataset processing jobs
                futures = []

                for worker_id, dataset_path in datasets_to_process:
                    future = executor.submit(
                        self.distribute_dataset_to_temp_chunks,
                        dataset_path,
                        worker_id,
                        self.num_chunks,
                        self.seed,
                        self.buffer_size,
                        self.progress_interval,
                        self.temp_dir,
                        flush_manager.flush_queue,
                        self.error_queue,
                        self.progress_queue,
                        self.val_fraction
                    )
                    futures.append(future)

                # Start progress monitoring in background
                import threading
                progress_thread = threading.Thread(
                    target=self._monitor_progress,
                    args=(self.progress_queue, len(datasets_to_process))
                )
                progress_thread.start()

                # Collect results and check for errors
                distribution_results = []
                last_status_check = time.time()

                for future in concurrent.futures.as_completed(futures):
                    # Check for flush errors
                    flush_manager.check_errors()

                    # Periodically check flush manager status
                    if time.time() - last_status_check > 10:  # Check every 10 seconds
                        status = flush_manager.get_status()
                        logger.debug(f"Flush manager status: {status['alive_writers']}/{status['total_writers']} "
                                   f"writers alive, queue size: {status['queue_size']}")

                        if status['dead_writers']:
                            logger.error(f"Flush writers {status['dead_writers']} have died unexpectedly!")
                            # Check for error messages
                            flush_manager.check_errors()
                            raise RuntimeError(f"Flush writers {status['dead_writers']} died")

                        if status['queue_size'] > 800:  # Warn if queue is getting full (>80%)
                            logger.warning(f"Flush queue is {status['queue_size']}/1000 - writers may be struggling")

                        last_status_check = time.time()

                    # Legacy check (kept for backwards compatibility)
                    dead_writers = [i for i, p in enumerate(flush_manager.writer_processes)
                                  if not p.is_alive()]
                    if dead_writers:
                        logger.error(f"Flush writers {dead_writers} have died unexpectedly!")
                        # Check for error messages
                        flush_manager.check_errors()

                    try:
                        result = future.result()
                        distribution_results.append(result)
                    except Exception as e:
                        # Log the exception with full traceback
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
                        logger.error(f"Worker failed with exception:\n{tb_str}")

                        # Cancel remaining tasks
                        for f in futures:
                            f.cancel()
                        raise

                # Wait for progress monitoring to complete
                progress_thread.join()

            # Wait for all flush operations to complete
            logger.info("Waiting for flush operations to complete...")
            time.sleep(2.0)  # Give flush processes time to complete
            flush_manager.check_errors()

            # Stop flush manager
            flush_manager.stop()
            flush_manager = None

            logger.info("Dataset distribution complete")

            # Step 4: Shuffle and finalize chunks using processes
            logger.info("=== STEP 4: FINALIZING CHUNKS ===")

            # Scan for existing valid output chunks
            finalization_results = []
            chunks_to_process = []

            if self.resume and not self.force_reprocess:
                try:
                    logger.info("Scanning for existing output chunks...")
                    completed_output_chunks = self._scan_output_chunks(common_dtype)

                    # Determine which chunks need processing
                    for chunk_id in range(self.num_chunks):
                        if not completed_output_chunks.get(chunk_id, False):
                            chunks_to_process.append(chunk_id)
                        else:
                            # Read existing chunk info to add to results
                            chunk_prefix = self.chunks_dir / f"chunk_{chunk_id:04d}"
                            idx_path = f"{chunk_prefix}.idx"

                            try:
                                with open(idx_path, 'rb') as f:
                                    f.seek(9 + 8 + 1)  # Skip header, version, dtype_code
                                    sequence_count = struct.unpack("<Q", f.read(8))[0]

                                finalization_results.append({
                                    'chunk_id': chunk_id,
                                    'sequences': sequence_count,
                                })
                            except Exception as e:
                                logger.warning(f"Failed to read existing chunk {chunk_id}: {e}")
                                # If we can't read it, reprocess it
                                chunks_to_process.append(chunk_id)

                    num_existing = len(finalization_results)
                    logger.info(f"Found {num_existing} existing valid chunks")
                    logger.info(f"Will process {len(chunks_to_process)} chunks")

                except Exception as e:
                    logger.warning(f"Failed to scan output chunks: {e}")
                    logger.info("Will process all chunks")
                    chunks_to_process = list(range(self.num_chunks))
            else:
                # Resume disabled or force_reprocess enabled
                chunks_to_process = list(range(self.num_chunks))

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for chunk_id in chunks_to_process:
                    future = executor.submit(
                        self.shuffle_and_finalize_chunk,
                        chunk_id,
                        self.temp_dir,
                        self.chunks_dir,
                        self.seed,
                        len(self.input_paths)
                    )
                    futures.append(future)

                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(chunks_to_process),
                                   desc="Finalizing chunks"):
                    result = future.result()
                    finalization_results.append(result)

            # Finalize validation set if enabled
            val_result = None
            if self.val_fraction > 0:
                logger.info("=== STEP 4b: FINALIZING VALIDATION SET ===")
                val_result = self.shuffle_and_finalize_chunk(
                    chunk_id=self.num_chunks,  # validation uses chunk_id = num_chunks
                    temp_dir=self.temp_dir,
                    output_dir=self.output_dir,
                    seed=self.seed,
                    num_workers=len(self.input_paths),
                    output_prefix="validation"
                )
                logger.info(f"Validation set: {val_result['sequences']} sequences")

            # Step 5: Write metadata
            logger.info("=== STEP 5: WRITING METADATA ===")
            self._write_process_metadata(validation_results, distribution_results, finalization_results, val_result)

            # Step 6: Validate output
            validation_passed = self._validate_output(validation_results, finalization_results, val_result)

            # Step 7: Cleanup temp directory if validation passed
            if validation_passed:
                logger.info("=== STEP 6: CLEANUP ===")
                if self.temp_dir.exists():
                    logger.info(f"Removing temporary directory: {self.temp_dir}")
                    shutil.rmtree(self.temp_dir)
                    logger.info("✓ Temporary directory cleaned up")
            else:
                logger.warning("⚠ Validation failed - temporary directory NOT removed for debugging")
                logger.warning(f"  Temp directory: {self.temp_dir}")

            # Final summary
            total_chunk_sequences = sum(r['sequences'] for r in finalization_results)
            val_sequences = val_result['sequences'] if val_result else 0
            total_final_sequences = total_chunk_sequences + val_sequences
            non_empty_chunks = sum(1 for r in finalization_results if r['sequences'] > 0)

            logger.info(f"=== CHUNKING COMPLETE ===")
            logger.info(f"  Training sequences: {total_chunk_sequences}")
            if val_result:
                logger.info(f"  Validation sequences: {val_sequences}")
                actual_val_pct = 100.0 * val_sequences / total_final_sequences if total_final_sequences > 0 else 0
                logger.info(f"  Validation percentage: {actual_val_pct:.2f}% (requested: {self.val_fraction * 100:.1f}%)")
                logger.info(f"  Chunks directory: {self.chunks_dir}")
            logger.info(f"  Non-empty chunks: {non_empty_chunks}/{self.num_chunks}")
            logger.info(f"  Output directory: {self.output_dir}")

            # Verify sequence count consistency
            original_sequences = sum(v['sequence_count'] for v in validation_results)
            if total_final_sequences != original_sequences:
                logger.warning(f"Sequence count mismatch: {original_sequences} -> {total_final_sequences}")
            else:
                logger.info("✓ Sequence count verified")
                
            if not validation_passed:
                raise RuntimeError("Output validation failed")

        except Exception as e:
            logger.error(f"Processing failed: {e}")

            # Cleanup flush manager on failure
            if flush_manager:
                try:
                    flush_manager.stop()
                except:
                    pass

            # Check for errors in queues
            try:
                while True:
                    error = self.error_queue.get_nowait()
                    if isinstance(error, dict):
                        if 'worker_id' in error:
                            logger.error(f"Worker {error['worker_id']} error details:")
                            logger.error(f"  Dataset: {error['dataset_path']}")
                            logger.error(f"  Exception type: {error['exception_type']}")
                            logger.error(f"  Exception message: {error['exception_message']}")
                            logger.error(f"  Traceback:\n{error['traceback']}")
                        elif 'flush_writer_id' in error:
                            logger.error(f"Flush writer {error['flush_writer_id']} error details:")
                            logger.error(f"  Exception type: {error['exception_type']}")
                            logger.error(f"  Exception message: {error['exception_message']}")
                            logger.error(f"  Traceback:\n{error['traceback']}")
                        else:
                            logger.error(f"Unknown error format: {error}")
                    else:
                        logger.error(f"Worker error: {error}")
            except queue.Empty:
                pass

            # Don't cleanup on failure - keep temp directory for debugging
            logger.error(f"⚠ Temporary directory preserved for debugging: {self.temp_dir}")
            raise

    def _write_process_metadata(self,
                                validation_results: List[Dict],
                                distribution_results: List[Dict],
                                finalization_results: List[Dict],
                                val_result: Optional[Dict] = None) -> None:
        """Write comprehensive metadata about the chunking process"""
        metadata_path = self.output_dir / "chunking_metadata.txt"

        with open(metadata_path, 'w') as f:
            f.write("Streaming Dataset Chunking Metadata (OPTIMIZED)\n")
            f.write("==============================================\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Number of chunks: {self.num_chunks}\n")
            f.write(f"  Shuffle seed: {self.seed}\n")
            f.write(f"  Number of workers: {self.num_workers}\n")
            f.write(f"  Number of flush writers: {self.num_flush_writers}\n")
            f.write(f"  Input datasets: {len(self.input_paths)}\n")
            f.write(f"  Buffer size: {self.buffer_size}\n")
            f.write(f"  Progress interval: {self.progress_interval}\n")
            if self.val_fraction > 0:
                f.write(f"  Validation split: {self.val_fraction * 100:.1f}%\n")
            f.write("\n")

            f.write("Input datasets:\n")
            for validation in validation_results:
                f.write(f"  {validation['path_prefix']}: "
                        f"{validation['sequence_count']} sequences, "
                        f"{validation['bin_size_bytes'] / 1024 ** 3:.2f} GB\n")
            f.write("\n")

            f.write("Final chunk statistics:\n")
            non_empty = sum(1 for r in finalization_results if r['sequences'] > 0)
            chunk_sequences = sum(r['sequences'] for r in finalization_results)
            f.write(f"  Non-empty chunks: {non_empty}/{self.num_chunks}\n")
            f.write(f"  Empty chunks: {self.num_chunks - non_empty}\n")
            f.write(f"  Training sequences: {chunk_sequences}\n")
            if self.val_fraction > 0:
                chunks_bin_files = list(self.chunks_dir.glob("chunk_*.bin"))
                chunks_size_gb = sum(f.stat().st_size for f in chunks_bin_files) / (1024**3)
                f.write(f"  Training size: {chunks_size_gb:.2f} GB\n")
            f.write("\n")

            # Validation set statistics
            if val_result:
                val_sequences = val_result['sequences']
                f.write("Validation set statistics:\n")
                f.write(f"  Validation sequences: {val_sequences}\n")
                val_bin = self.output_dir / "validation.bin"
                if val_bin.exists():
                    val_size_gb = val_bin.stat().st_size / (1024**3)
                    f.write(f"  Validation size: {val_size_gb:.4f} GB\n")
                total = chunk_sequences + val_sequences
                actual_pct = 100.0 * val_sequences / total if total > 0 else 0
                f.write(f"  Actual validation percentage: {actual_pct:.2f}%\n")
                f.write(f"  Requested validation percentage: {self.val_fraction * 100:.1f}%\n")
                f.write("\n")

            # Data integrity verification
            original_sequences = sum(v['sequence_count'] for v in validation_results)
            val_sequences = val_result['sequences'] if val_result else 0
            final_sequences = chunk_sequences + val_sequences
            f.write(f"Data integrity:\n")
            f.write(f"  Original sequences: {original_sequences}\n")
            f.write(f"  Training sequences: {chunk_sequences}\n")
            if val_result:
                f.write(f"  Validation sequences: {val_sequences}\n")
            f.write(f"  Total sequences: {final_sequences}\n")
            f.write(f"  Integrity check: {'PASS' if original_sequences == final_sequences else 'FAIL'}\n")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        # Only cleanup if processing failed - successful runs handle cleanup in process()
        if exc_type is not None:
            logger.warning("Processing failed - preserving temp directory for debugging")


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient dataset chunker (OPTIMIZED)")

    # Input specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-paths", nargs="+",
                             help="List of input dataset path prefixes")
    input_group.add_argument("--input-folder",
                             help="Single folder containing idx/bin dataset pairs")
    input_group.add_argument("--input-folders", nargs="+",
                             help="Multiple folders containing idx/bin dataset pairs")

    parser.add_argument("--output-dir", required=True,
                        help="Output directory for chunks")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate datasets without chunking")

    parser.add_argument("--num-chunks", type=int, default=4096,
                        help="Number of chunks to create")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--num-flush-writers", type=int, default=4,
                        help="Number of async flush writer processes")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for shuffling")
    parser.add_argument("--temp-dir", default=None,
                        help="Temporary directory (default: auto-created)")
    parser.add_argument("--buffer-size", type=int, default=500,
                        help="Number of sequences to buffer before flushing")
    parser.add_argument("--progress-interval", type=int, default=500000,
                        help="Update progress every N sequences (default: 500000)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Enable automatic resume from interrupted runs (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Disable resume functionality")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Clear temp and output directories, start fresh")
    parser.add_argument("--val-percent", type=float, default=0.0,
                        help="Hold out X%% of sequences as validation set (default: 0 = disabled, e.g. 1.0 for 1%%)")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Create chunker using context manager for proper cleanup
        with StreamingDatasetChunker(
                input_paths=args.input_paths,
                input_folder=args.input_folder,
                input_folders=args.input_folders,
                output_dir=args.output_dir,
                num_chunks=args.num_chunks,
                num_workers=args.num_workers,
                num_flush_writers=args.num_flush_writers,
                seed=args.seed,
                temp_dir=args.temp_dir,
                buffer_size=args.buffer_size,
                progress_interval=args.progress_interval,
                resume=args.resume,
                force_reprocess=args.force_reprocess,
                val_percent=args.val_percent,
        ) as chunker:

            if args.validate_only:
                # Only run validation
                logger.info("Running validation only...")
                for path in chunker.input_paths:
                    validation_info = chunker.validate_dataset_pair(path)
                    logger.info(f"✓ {path}: {validation_info['sequence_count']} sequences, "
                                f"{validation_info['bin_size_bytes'] / 1024 ** 3:.2f} GB")

                # Also validate dtype consistency
                validation_results = [chunker.validate_dataset_pair(path) for path in chunker.input_paths]
                common_dtype = chunker.validate_dtype_consistency(validation_results)
                logger.info(f"✓ Dtype consistency validated: {common_dtype}")
                logger.info("All datasets validated successfully!")
            else:
                # Run full chunking process
                chunker.process()

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()