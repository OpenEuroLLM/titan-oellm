"""
Deterministic packed dataset with greedy packing.

Guarantees identical packed sequences regardless of dp_world_size,
local_batch_size, or gradient_accumulation_steps — given the same
global_batch_size and seed.

Combines document-level determinism with global sequence packing,
eliminating the non-determinism of per-rank StreamingSequencer.

Greedy packing: continuous token stream cut at fixed positions.
Sequences computed on-the-fly via binary search — no stored index.

For best-fit bin-packing (reduced token waste), see BestFitPackedDataset.

Design:
  - Hierarchical index: per-chunk cumulative token counts (32 KB) +
    per-document lengths from .idx files (lazy, mmap'd).
  - Checkpointing is trivial: just one integer (global_sequence_id).
  - Scales to 1-10T tokens, 100-4000 GPUs, global_batch_size up to 16M.

Batch diversity via strided assignment:
  Step S's global batch = {S + k*E : k=0..GBS-1} where E = steps_per_epoch.
  Each rank reads SPR "lanes", each advancing by 1 sequence per step.
  This spreads every batch across the full epoch (different chunks),
  while each lane reads sequentially through its region of the stream.
  The batch composition depends only on GBS, not dp_world_size.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy
import torch

from titan_oellm.datasets.dataloader.mmap_dataset_chunked import (
    _IndexReader,
    _RandomAccessBinReader,
)

logger = logging.getLogger(__name__)


class DeterministicPackedDataset(torch.utils.data.IterableDataset):
    """
    Deterministic packed dataset that yields fixed-length token sequences
    using greedy packing.

    All ranks compute the same global chunk permutation and cumulative token
    counts per epoch. Sequences are defined as contiguous token ranges in the
    global token stream. Each rank reads a deterministic slice of global
    sequences, ensuring node-count independence.

    Args:
        chunks_dir: Directory containing chunk_*.bin / chunk_*.idx files.
        dp_world_size: Number of data-parallel ranks.
        dp_rank: This rank's data-parallel index.
        global_batch_size: Number of packed sequences per training step (global).
        seq_len: Target sequence length (each output has seq_len + 1 tokens).
        min_sequence_length: Skip documents shorter than this (default: 0).
        eos_id: EOS token ID to insert between documents (None = no separator).
        infinite: Loop over epochs (default: True).
        seed: Random seed for chunk permutation (default: 1).
        exclude_first_n_per_chunk: Training split — skip first N docs per chunk.
        use_only_first_n_per_chunk: Validation split — use only first N docs.
    """

    def __init__(
        self,
        chunks_dir: str | list[str],
        dp_world_size: int,
        dp_rank: int,
        global_batch_size: int,
        seq_len: int,
        min_sequence_length: int = 0,
        eos_id: Optional[int] = None,
        infinite: bool = True,
        seed: int = 1,
        exclude_first_n_per_chunk: Optional[int] = None,
        use_only_first_n_per_chunk: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(chunks_dir, list):
            self.chunks_dirs = [Path(d) for d in chunks_dir]
        else:
            self.chunks_dirs = [Path(chunks_dir)]
        self.chunks_dir = self.chunks_dirs[0]  # kept for logging / back-compat
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.tokens_per_seq = seq_len + 1  # +1 for input/target offset
        self.min_sequence_length = min_sequence_length
        self.eos_id = eos_id
        self.has_eos = eos_id is not None
        self.infinite = infinite
        self.seed = seed

        # Split params
        self.exclude_first_n_per_chunk = exclude_first_n_per_chunk
        self.use_only_first_n_per_chunk = use_only_first_n_per_chunk

        assert global_batch_size % dp_world_size == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"dp_world_size ({dp_world_size})"
        )
        self.sequences_per_rank = global_batch_size // dp_world_size

        # Discover all chunks
        self.all_chunks = self._get_chunk_prefixes()
        num_chunks = len(self.all_chunks)
        dirs_str = self.chunks_dirs[0] if len(self.chunks_dirs) == 1 else self.chunks_dirs
        logger.info(f"Found {num_chunks} total chunks in {dirs_str}")

        # Read per-chunk document lengths and compute effective token counts
        self._all_chunk_doc_lengths = self._read_all_chunk_doc_lengths()

        # Setup global ordering for epoch 0
        self.epoch_counter = 0
        self._setup_global_ordering(self.epoch_counter)

        # Position tracking — the entire state is this one number
        self.global_sequence_id = 0
        self.sample_counter = 0

        # Currently loaded chunk(s) and their cached data.
        # With strided batch assignment, each rank reads from up to SPR
        # different chunks per step — cache must accommodate all of them.
        self._loaded_chunks: Dict[int, _ChunkData] = {}
        self._max_loaded_chunks = max(8, self.sequences_per_rank + 2)

        # Log info
        split_info = ""
        if exclude_first_n_per_chunk:
            split_info = f" (training: skip first {exclude_first_n_per_chunk} docs/chunk)"
        elif use_only_first_n_per_chunk:
            split_info = f" (validation: only first {use_only_first_n_per_chunk} docs/chunk)"

        logger.info(
            f"DeterministicPackedDataset: rank {dp_rank}/{dp_world_size}, "
            f"global_bs={global_batch_size}, seqs_per_rank={self.sequences_per_rank}, "
            f"total_effective_tokens={self.total_effective_tokens}, "
            f"total_sequences={self.total_sequences}, "
            f"steps_per_epoch={self.steps_per_epoch}, "
            f"seq_len={seq_len}, min_seq_len={min_sequence_length}, "
            f"eos={'yes' if self.has_eos else 'no'}, "
            f"packing=greedy, "
            f"chunks={num_chunks}{split_info}"
        )

    # ── Chunk discovery ──────────────────────────────────────────────────

    def _get_chunk_prefixes(self) -> List[str]:
        """Discover all available chunk files across all chunks_dirs, sorted for deterministic ordering."""
        all_chunks = []

        for chunks_dir in self.chunks_dirs:
            if not chunks_dir.exists():
                raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

            for idx_file in chunks_dir.glob("chunk_*.idx"):
                chunk_prefix = str(idx_file).replace('.idx', '')
                bin_file = Path(f"{chunk_prefix}.bin")
                if bin_file.exists():
                    all_chunks.append(chunk_prefix)
                else:
                    logger.warning(f"Missing .bin file for {idx_file}")

        all_chunks.sort()

        if not all_chunks:
            raise RuntimeError(f"No valid chunk pairs found in {self.chunks_dirs}")
        return all_chunks

    # ── Per-chunk metadata ───────────────────────────────────────────────

    def _read_all_chunk_doc_lengths(self) -> List[np.ndarray]:
        """Read document lengths from every chunk's .idx, applying split mode.

        Returns list of arrays, one per chunk. Each array contains the effective
        document lengths (after split and min_length filtering), with EOS token
        accounted for.
        """
        all_lengths = []
        for chunk_id, chunk_path in enumerate(self.all_chunks):
            try:
                idx_reader = _IndexReader(f"{chunk_path}.idx")
                raw_lengths = np.array(idx_reader.sequence_lengths, dtype=np.int64)
                raw_count = len(raw_lengths)

                # Apply split mode
                start = 0
                end = raw_count
                if self.exclude_first_n_per_chunk is not None:
                    start = min(self.exclude_first_n_per_chunk, raw_count)
                elif self.use_only_first_n_per_chunk is not None:
                    end = min(self.use_only_first_n_per_chunk, raw_count)

                effective_lengths = raw_lengths[start:end].copy()

                # Filter by min_sequence_length
                if self.min_sequence_length > 0:
                    mask = effective_lengths >= self.min_sequence_length
                    effective_lengths = effective_lengths[mask]

                # Add EOS token to each document's length
                if self.has_eos:
                    effective_lengths = effective_lengths + 1

                all_lengths.append(effective_lengths)

                del idx_reader
            except Exception as e:
                logger.error(f"Failed to read chunk {chunk_path}: {e}")
                raise

        return all_lengths

    # ── Global ordering (per epoch) ──────────────────────────────────────

    def _setup_global_ordering(self, epoch: int) -> None:
        """Compute deterministic chunk permutation and cumulative token counts.

        All ranks call this with the same epoch and seed, producing identical
        results.
        """
        rng = np.random.RandomState(self.seed + epoch)
        num_chunks = len(self.all_chunks)
        self.chunk_order = rng.permutation(num_chunks).tolist()

        # Compute per-chunk effective token counts in permuted order
        ordered_token_counts = []
        for chunk_id in self.chunk_order:
            total_tokens = int(np.sum(self._all_chunk_doc_lengths[chunk_id]))
            ordered_token_counts.append(total_tokens)

        # Cumulative token counts: cum_chunk_tokens[i] = total tokens through
        # permuted chunks 0..i-1. cum_chunk_tokens[0] = 0.
        self.cum_chunk_tokens = np.zeros(num_chunks + 1, dtype=np.int64)
        for i, count in enumerate(ordered_token_counts):
            self.cum_chunk_tokens[i + 1] = self.cum_chunk_tokens[i] + count

        self.total_effective_tokens = int(self.cum_chunk_tokens[-1])
        self.total_sequences = self.total_effective_tokens // self.tokens_per_seq
        self.steps_per_epoch = self.total_sequences // self.global_batch_size

        # Build reverse lookup
        self._chunk_order_index = {}
        for order_idx, chunk_id in enumerate(self.chunk_order):
            self._chunk_order_index[chunk_id] = order_idx

    # ── Chunk loading with cache ─────────────────────────────────────────

    def _get_chunk_data(self, chunk_id: int) -> '_ChunkData':
        """Get loaded chunk data, loading if necessary. FIFO cache of loaded chunks."""
        if chunk_id in self._loaded_chunks:
            return self._loaded_chunks[chunk_id]

        # Evict oldest if at capacity
        if len(self._loaded_chunks) >= self._max_loaded_chunks:
            oldest_key = next(iter(self._loaded_chunks))
            old = self._loaded_chunks.pop(oldest_key)
            old.close()

        chunk_data = _ChunkData(
            chunk_id=chunk_id,
            chunk_path=self.all_chunks[chunk_id],
            effective_doc_lengths=self._all_chunk_doc_lengths[chunk_id],
            exclude_first_n=self.exclude_first_n_per_chunk,
            use_only_first_n=self.use_only_first_n_per_chunk,
            min_sequence_length=self.min_sequence_length,
            has_eos=self.has_eos,
        )
        self._loaded_chunks[chunk_id] = chunk_data
        return chunk_data

    # ── Token position → chunk/doc mapping ───────────────────────────────

    def _token_pos_to_chunk_and_offset(self, global_token_pos: int) -> Tuple[int, int]:
        """Map a global token position to (chunk_id, offset_within_chunk_tokens).

        Uses binary search on cum_chunk_tokens.
        """
        chunk_order_idx = int(
            np.searchsorted(self.cum_chunk_tokens[1:], global_token_pos, side='right')
        )
        chunk_id = self.chunk_order[chunk_order_idx]
        offset_in_chunk = int(global_token_pos - self.cum_chunk_tokens[chunk_order_idx])
        return chunk_id, offset_in_chunk

    # ── Read a packed sequence ───────────────────────────────────────────

    def _read_packed_sequence(self, global_token_start: int) -> Dict[str, Any]:
        """Read tokens_per_seq tokens starting at global_token_start.

        Reads across document and chunk boundaries as needed. Returns a dict
        compatible with the collator: {'tokens': [np.array], 'seqlen': [...]}.
        """
        tokens_needed = self.tokens_per_seq
        all_tokens = []
        seqlens = []
        tokens_collected = 0

        current_token_pos = global_token_start

        while tokens_collected < tokens_needed:
            # Find which chunk and offset
            chunk_id, offset_in_chunk = self._token_pos_to_chunk_and_offset(
                current_token_pos
            )
            chunk_data = self._get_chunk_data(chunk_id)

            # Find which document within this chunk contains the offset
            doc_idx, token_in_doc = chunk_data.offset_to_doc_and_token(offset_in_chunk)

            if doc_idx >= len(chunk_data.effective_doc_lengths):
                # Past end of this chunk — advance to next in permuted order
                chunk_order_idx = self._chunk_order_index[chunk_id]
                next_order_idx = chunk_order_idx + 1
                if next_order_idx >= len(self.chunk_order):
                    # End of epoch token stream
                    break
                current_token_pos = int(self.cum_chunk_tokens[next_order_idx])
                continue

            # How many tokens remain in this document?
            doc_effective_len = int(chunk_data.effective_doc_lengths[doc_idx])
            tokens_remaining_in_doc = doc_effective_len - token_in_doc

            # How many tokens we want from this document
            tokens_to_take = min(tokens_remaining_in_doc, tokens_needed - tokens_collected)

            # Read the actual tokens from the binary file
            raw_doc_idx = chunk_data.effective_to_raw_idx[doc_idx]
            raw_pointer, raw_length, _ = chunk_data.index_reader[raw_doc_idx]

            # Compute token range accounting for EOS
            if self.has_eos:
                raw_doc_tokens = raw_length  # tokens in .bin file (no EOS)
                # The effective doc is: [raw_tokens..., EOS]
                # token_in_doc indexes into this effective sequence
                eos_in_range_start = token_in_doc
                eos_in_range_end = token_in_doc + tokens_to_take

                # How many of the tokens we read are from the raw file vs EOS?
                raw_start = min(eos_in_range_start, raw_doc_tokens)
                raw_end = min(eos_in_range_end, raw_doc_tokens)
                raw_count = raw_end - raw_start

                if raw_count > 0:
                    dtype_size = np.dtype(chunk_data.index_reader.dtype).itemsize
                    read_offset = raw_pointer + raw_start * dtype_size
                    doc_tokens = chunk_data.bin_reader.read(
                        dtype=chunk_data.index_reader.dtype,
                        count=raw_count,
                        offset=read_offset,
                    )
                    all_tokens.append(doc_tokens)

                # Append EOS if the range includes the EOS position
                if eos_in_range_end > raw_doc_tokens:
                    eos_count = min(
                        eos_in_range_end - max(eos_in_range_start, raw_doc_tokens),
                        1  # only one EOS per document
                    )
                    if eos_count > 0:
                        all_tokens.append(
                            np.array([self.eos_id], dtype=chunk_data.index_reader.dtype)
                        )
            else:
                # No EOS — straightforward read
                dtype_size = np.dtype(chunk_data.index_reader.dtype).itemsize
                read_offset = raw_pointer + token_in_doc * dtype_size
                doc_tokens = chunk_data.bin_reader.read(
                    dtype=chunk_data.index_reader.dtype,
                    count=tokens_to_take,
                    offset=read_offset,
                )
                all_tokens.append(doc_tokens)

            # Track seqlens — this is the length contribution from this document
            # into the current sequence. Seqlens marks document boundaries for
            # FlexAttention/block_causal masking.
            if (seqlens and token_in_doc > 0
                    and tokens_collected > 0):
                # Continuing a document from the previous sequence boundary —
                # this is a document continuation, extend the last seqlen
                seqlens[-1] += tokens_to_take
            else:
                seqlens.append(tokens_to_take)

            tokens_collected += tokens_to_take
            current_token_pos += tokens_to_take

        # Concatenate all token arrays
        if all_tokens:
            sequence = np.concatenate(all_tokens)
        else:
            sequence = np.array([], dtype=np.int64)

        # Adjust last seqlen if total doesn't match exactly
        total_seqlen = sum(seqlens) if seqlens else 0
        if total_seqlen != len(sequence) and seqlens:
            logger.warning(
                f"Seqlen fixup: sum(seqlens)={total_seqlen} != len(sequence)={len(sequence)}, "
                f"adjusting last seqlen by {total_seqlen - len(sequence)}"
            )
            seqlens[-1] = seqlens[-1] - (total_seqlen - len(sequence))

        return {
            'tokens': [sequence],
            'seqlen': seqlens,
        }

    # ── Iterator protocol ────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        # Strided batch assignment for diversity + dp_world_size independence.
        #
        # The global batch at step S consists of GBS sequences evenly spaced
        # across the epoch: {S + k*E : k = 0, ..., GBS-1} where E = steps_per_epoch.
        # This set is identical regardless of dp_world_size (depends only on GBS).
        #
        # Rank R reads SPR of these (its "lanes"), each advancing by 1 per step.
        # Over time each lane reads sequentially through its region of the stream.
        local_offset = self.global_sequence_id % self.sequences_per_rank
        step = self.global_sequence_id // self.sequences_per_rank

        if local_offset == 0 and step >= self.steps_per_epoch:
            if self.infinite:
                self.epoch_counter += 1
                self._setup_global_ordering(self.epoch_counter)
                self.global_sequence_id = 0
                step = 0
                local_offset = 0
                logger.info(
                    f"Rank {self.dp_rank}: starting epoch {self.epoch_counter}"
                )
            else:
                raise StopIteration

        # Lane index within the global batch: [0, GBS)
        batch_position = self.dp_rank * self.sequences_per_rank + local_offset
        # Map to global sequence via stride: each lane is spaced E apart
        global_seq = step + batch_position * self.steps_per_epoch
        token_pos = global_seq * self.tokens_per_seq

        # Read the packed sequence
        sample = self._read_packed_sequence(token_pos)

        self.global_sequence_id += 1
        self.sample_counter += 1

        return sample

    def __len__(self) -> int:
        """Total packed sequences this rank will yield per epoch."""
        return self.steps_per_epoch * self.sequences_per_rank

    # ── Checkpointing ────────────────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle state without file handles."""
        state = self.__dict__.copy()
        # Remove loaded chunk data (has file handles)
        state.pop('_loaded_chunks', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and reinitialize."""
        self.__dict__.update(state)
        self._loaded_chunks = {}

    # ── Stateful protocol (for StatefulDataLoader checkpoint support) ────

    def state_dict(self) -> Dict[str, Any]:
        """Save minimal state for fast checkpoint resume."""
        return {
            "global_sequence_id": self.global_sequence_id,
            "epoch_counter": self.epoch_counter,
            "sample_counter": self.sample_counter,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore position from checkpoint — no fast-forwarding needed."""
        self.epoch_counter = state_dict["epoch_counter"]
        self.global_sequence_id = state_dict["global_sequence_id"]
        self.sample_counter = state_dict.get("sample_counter", 0)
        # Re-setup global ordering for the restored epoch
        self._setup_global_ordering(self.epoch_counter)

    # ── Stats / info ─────────────────────────────────────────────────────

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'worker_rank': self.dp_rank,
            'dp_world_size': self.dp_world_size,
            'global_batch_size': self.global_batch_size,
            'sequences_per_rank': self.sequences_per_rank,
            'total_effective_tokens': self.total_effective_tokens,
            'total_sequences': self.total_sequences,
            'steps_per_epoch': self.steps_per_epoch,
            'epoch_counter': self.epoch_counter,
            'global_sequence_id': self.global_sequence_id,
            'sample_counter': self.sample_counter,
            'seq_len': self.seq_len,
            'min_sequence_length': self.min_sequence_length,
            'packing_mode': 'greedy',
            'num_chunks': len(self.all_chunks),
            'loaded_chunks': list(self._loaded_chunks.keys()),
        }

    def __del__(self) -> None:
        if hasattr(self, '_loaded_chunks'):
            for chunk_data in self._loaded_chunks.values():
                chunk_data.close()
            self._loaded_chunks.clear()


class _ChunkData:
    """Cached data for a loaded chunk — index reader, bin reader, and
    precomputed effective document mappings."""

    def __init__(
        self,
        chunk_id: int,
        chunk_path: str,
        effective_doc_lengths: np.ndarray,
        exclude_first_n: Optional[int],
        use_only_first_n: Optional[int],
        min_sequence_length: int,
        has_eos: bool,
    ):
        self.chunk_id = chunk_id
        self.index_reader = _IndexReader(f"{chunk_path}.idx")
        self.bin_reader = _RandomAccessBinReader(f"{chunk_path}.bin")
        self.effective_doc_lengths = effective_doc_lengths

        # Build mapping from effective doc index → raw doc index in .idx file
        raw_count = len(self.index_reader)
        start = 0
        end = raw_count
        if exclude_first_n is not None:
            start = min(exclude_first_n, raw_count)
        elif use_only_first_n is not None:
            end = min(use_only_first_n, raw_count)

        raw_indices = np.arange(start, end, dtype=np.int64)

        # Filter by min_sequence_length (same filter as in _read_all_chunk_doc_lengths)
        if min_sequence_length > 0:
            raw_lengths = np.array(
                [self.index_reader.sequence_lengths[i] for i in raw_indices],
                dtype=np.int64,
            )
            mask = raw_lengths >= min_sequence_length
            raw_indices = raw_indices[mask]

        self.effective_to_raw_idx = raw_indices

        # Cumulative effective token counts within this chunk
        # cum_doc_tokens[i] = total effective tokens through docs 0..i-1
        self.cum_doc_tokens = np.zeros(len(effective_doc_lengths) + 1, dtype=np.int64)
        np.cumsum(effective_doc_lengths, out=self.cum_doc_tokens[1:])

    def offset_to_doc_and_token(self, offset_in_chunk: int) -> Tuple[int, int]:
        """Map a token offset within this chunk to (effective_doc_idx, token_in_doc)."""
        if offset_in_chunk >= self.cum_doc_tokens[-1]:
            return len(self.effective_doc_lengths), 0

        doc_idx = int(
            np.searchsorted(self.cum_doc_tokens[1:], offset_in_chunk, side='right')
        )
        token_in_doc = int(offset_in_chunk - self.cum_doc_tokens[doc_idx])
        return doc_idx, token_in_doc

    def close(self):
        if hasattr(self, 'index_reader'):
            del self.index_reader
        if hasattr(self, 'bin_reader'):
            del self.bin_reader
