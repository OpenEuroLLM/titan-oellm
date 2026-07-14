"""
Best-fit bin-packing dataset with pre-computed packing plan.

Guarantees identical packed sequences regardless of dp_world_size,
local_batch_size, or gradient_accumulation_steps — given the same
global_batch_size and seed.

Uses a C++ pybind11 extension (bestfit_packer) to build a packing plan
that combines document fragments into fixed-length sequences, reducing
token waste compared to greedy slicing. The plan is cached as .npz
and memory-mapped for fast reload.

Design:
  - Hierarchical index: per-chunk cumulative token counts (32 KB) +
    per-document lengths from .idx files (lazy, mmap'd).
  - Sequential I/O within chunks is preserved.
  - Checkpointing is trivial: just one integer (global_sequence_id).
  - Scales to 1-10T tokens, 100-4000 GPUs, global_batch_size up to 16M.
"""

import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy
import torch

from titan_oellm.datasets.dataloader.mmap_dataset_chunked import (
    _IndexReader,
    _RandomAccessBinReader,
)
from titan_oellm.datasets.dataloader._eos_utils import append_eos_if_missing

logger = logging.getLogger(__name__)

# Number of docs to probe per chunk when detecting whether the raw .bin
# already has EOS at doc boundaries. Homogeneous within a chunk is assumed.
_EOS_PROBE_DOCS = 32


class BestFitPackedDataset(torch.utils.data.IterableDataset):
    """
    Best-fit packed dataset that yields fixed-length token sequences.

    All ranks compute the same global chunk permutation and cumulative token
    counts per epoch. A pre-computed packing plan assigns document fragments
    to sequences, minimizing wasted tokens. Each rank reads a deterministic
    slice of global sequences, ensuring node-count independence.

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
        best_fit_buffer_size: BST buffer size for best-fit packing (default: 32).
    """

    def __init__(
        self,
        chunks_dir: str,
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
        best_fit_buffer_size: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.chunks_dir = Path(chunks_dir)
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
        self.best_fit_buffer_size = best_fit_buffer_size
        # Optional override for cache directory. When None, falls back to
        # <chunks_dir>/.packing_cache for backward compatibility. A shared
        # cache_dir (e.g. <cache_base>/bfp_packing_cache) lets prebuild on the
        # login node and the multi-node sbatch job reuse the same .npz files
        # without writing into the (potentially read-only) dataset dir.
        self.cache_dir = Path(cache_dir) if cache_dir else None

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
        logger.info(f"Found {num_chunks} total chunks in {self.chunks_dir}")

        # Read per-chunk document lengths and compute effective token counts
        self._all_chunk_doc_lengths = self._read_all_chunk_doc_lengths()

        # Setup global ordering for epoch 0
        self.epoch_counter = 0
        self._setup_global_ordering(self.epoch_counter)

        # Build or load the packing plan
        self._bestfit_seq_doc_counts = None
        self._bestfit_doc_refs = None
        self._bestfit_seq_offsets = None
        self._build_or_load_bestfit_plan(self.epoch_counter)

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
            f"BestFitPackedDataset: rank {dp_rank}/{dp_world_size}, "
            f"global_bs={global_batch_size}, seqs_per_rank={self.sequences_per_rank}, "
            f"total_effective_tokens={self.total_effective_tokens}, "
            f"total_sequences={self.total_sequences}, "
            f"steps_per_epoch={self.steps_per_epoch}, "
            f"seq_len={seq_len}, min_seq_len={min_sequence_length}, "
            f"eos={'yes' if self.has_eos else 'no'}, "
            f"packing=best_fit, buffer_size={best_fit_buffer_size}, "
            f"chunks={num_chunks}{split_info}"
        )

    # ── Chunk discovery ──────────────────────────────────────────────────

    def _get_chunk_prefixes(self) -> List[str]:
        """Discover all available chunk files, sorted for deterministic ordering."""
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {self.chunks_dir}")

        idx_files = list(self.chunks_dir.glob("chunk_*.idx"))
        all_chunks = []

        for idx_file in idx_files:
            chunk_prefix = str(idx_file).replace('.idx', '')
            bin_file = Path(f"{chunk_prefix}.bin")
            if bin_file.exists():
                all_chunks.append(chunk_prefix)
            else:
                logger.warning(f"Missing .bin file for {idx_file}")

        all_chunks.sort()

        if not all_chunks:
            raise RuntimeError(f"No valid chunk pairs found in {self.chunks_dir}")
        return all_chunks

    # ── Per-chunk metadata ───────────────────────────────────────────────

    def _probe_chunk_has_inline_eos(
        self,
        idx_reader: '_IndexReader',
        chunk_path: str,
        raw_indices_to_probe: List[int],
    ) -> bool:
        """Read the last token of a sample of docs to decide whether this
        chunk's .bin already terminates documents with `self.eos_id`.

        Returns True only when *all* probed docs end with the eos token —
        we treat a chunk as inline-EOS only if it is unambiguously so.
        """
        if not self.has_eos or not raw_indices_to_probe:
            return False

        bin_reader = _RandomAccessBinReader(f"{chunk_path}.bin")
        try:
            dtype = idx_reader.dtype
            dtype_size = np.dtype(dtype).itemsize
            for raw_idx in raw_indices_to_probe:
                pointer, length, _ = idx_reader[raw_idx]
                if length <= 0:
                    return False
                last_offset = pointer + (length - 1) * dtype_size
                last_token = bin_reader.read(dtype=dtype, count=1, offset=last_offset)
                if int(last_token[0]) != int(self.eos_id):
                    return False
            return True
        finally:
            del bin_reader

    def _read_all_chunk_doc_lengths(self) -> List[np.ndarray]:
        """Read document lengths from every chunk's .idx, applying split mode.

        Returns list of arrays, one per chunk. Each array contains the effective
        document lengths (after split and min_length filtering), with EOS token
        accounted for.

        Side-effect: populates `self._chunk_has_inline_eos[chunk_id]` so
        `_read_packed_sequence` can avoid double-EOS.
        """
        all_lengths = []
        self._chunk_has_inline_eos: List[bool] = []
        # Per-chunk inline-EOS state is a static property of the tokenized
        # chunks; load it from cache to skip the expensive per-rank .bin probe.
        cached_eos = self._load_eos_state_cache() if self.has_eos else None
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

                # Inline-EOS state: use the cache when available, else probe the
                # chunk's .bin (reading up to _EOS_PROBE_DOCS docs' last tokens).
                # Probe indices come from the post-split slice so they match the
                # docs we'll actually read.
                if cached_eos is not None:
                    chunk_inline_eos = cached_eos[chunk_id]
                else:
                    probe_count = min(_EOS_PROBE_DOCS, end - start)
                    probe_indices = list(range(start, start + probe_count))
                    chunk_inline_eos = self._probe_chunk_has_inline_eos(
                        idx_reader, chunk_path, probe_indices,
                    )
                self._chunk_has_inline_eos.append(chunk_inline_eos)

                # Budget +1 only when we will actually inject EOS at read time.
                if self.has_eos and not chunk_inline_eos:
                    effective_lengths = effective_lengths + 1

                all_lengths.append(effective_lengths)

                del idx_reader
            except Exception as e:
                logger.error(f"Failed to read chunk {chunk_path}: {e}")
                raise

        # Persist the probed inline-EOS state so subsequent runs (and every rank
        # after the single-process prebuild) skip the probe entirely.
        if self.has_eos and cached_eos is None:
            self._write_eos_state_cache(self._chunk_has_inline_eos)

        if self.has_eos:
            inline_count = sum(self._chunk_has_inline_eos)
            total = len(self._chunk_has_inline_eos)
            if 0 < inline_count < total:
                logger.warning(
                    f"BestFitPackedDataset: mixed EOS state across chunks "
                    f"({inline_count}/{total} chunks already terminate docs with "
                    f"eos_id={self.eos_id}). Per-chunk dedup will keep packing "
                    f"correct, but consider re-tokenizing for consistency."
                )
            elif inline_count == total and total > 0:
                logger.info(
                    f"BestFitPackedDataset: all {total} chunks already terminate "
                    f"docs with eos_id={self.eos_id}; skipping in-pipeline injection."
                )

        return all_lengths

    # ── Inline-EOS state cache ───────────────────────────────────────────
    # The inline-EOS probe reads up to _EOS_PROBE_DOCS docs' last token from
    # every chunk's .bin. On a 2048-chunk store over a contended shared FS that
    # is ~30 min PER RANK, repeated every job — yet the result is a static
    # property of the tokenized chunks. We cache it keyed on chunk file sizes
    # (cheap stat; detects re-tokenization) so the probe runs at most once: the
    # single-process BFP prebuild warms it, then all ranks/jobs just load it.
    _EOS_CACHE_VERSION = "v1"

    def _compute_eos_cache_key(self) -> str:
        """Key the inline-EOS cache on chunk identity + eos/probe/split params.

        Deliberately excludes the EOS state itself (that's what we're caching)
        and uses file sizes rather than reading content, so building the key is
        cheap (a stat per chunk) and a re-tokenized corpus invalidates it.
        """
        hasher = hashlib.md5()
        hasher.update(f"eosfmt={self._EOS_CACHE_VERSION}".encode())
        hasher.update(f"cd={str(self.chunks_dir.resolve())}".encode())
        hasher.update(f"eos={self.eos_id}".encode())
        hasher.update(f"probe={_EOS_PROBE_DOCS}".encode())
        hasher.update(f"exf={self.exclude_first_n_per_chunk}".encode())
        hasher.update(f"unf={self.use_only_first_n_per_chunk}".encode())
        for chunk_path in self.all_chunks:
            name = Path(chunk_path).name
            try:
                bsz = os.path.getsize(f"{chunk_path}.bin")
                isz = os.path.getsize(f"{chunk_path}.idx")
            except OSError:
                bsz = isz = -1
            hasher.update(f"{name}:{bsz}:{isz}".encode())
        return hasher.hexdigest()[:16]

    def _eos_cache_path(self) -> Path:
        return self._resolve_cache_dir() / f"eosstate_{self._compute_eos_cache_key()}.json"

    def _load_eos_state_cache(self) -> Optional[List[bool]]:
        """Return the cached per-chunk inline-EOS flags, or None on miss."""
        path = self._eos_cache_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            states = data.get("states")
            if (isinstance(states, list)
                    and len(states) == len(self.all_chunks)
                    and all(isinstance(s, bool) for s in states)):
                logger.info(
                    f"Loaded inline-EOS state from cache ({path.name}); "
                    f"skipping per-chunk .bin probe of {len(states)} chunks."
                )
                return states
            logger.warning(
                f"inline-EOS cache {path.name} shape mismatch "
                f"({len(states) if isinstance(states, list) else 'n/a'} vs "
                f"{len(self.all_chunks)} chunks); reprobing."
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"inline-EOS cache unreadable ({e!r}); reprobing.")
        return None

    def _write_eos_state_cache(self, states: List[bool]) -> None:
        """Atomically persist per-chunk inline-EOS flags (concurrent ranks write
        identical content; os.replace makes partial reads impossible)."""
        path = self._eos_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "version": self._EOS_CACHE_VERSION,
                        "eos_id": self.eos_id,
                        "probe_docs": _EOS_PROBE_DOCS,
                        "states": [bool(s) for s in states],
                    },
                    f,
                )
            os.replace(tmp, path)
            logger.info(f"Cached inline-EOS state to {path.name}")
        except OSError as e:
            logger.warning(f"Could not write inline-EOS cache ({e!r}); continuing.")

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
        """Get loaded chunk data, loading if necessary. LRU cache of loaded chunks."""
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

    # ── Best-fit packing plan ───────────────────────────────────────────

    @staticmethod
    def _compile_bestfit_packer():
        """Compile and import the C++ bestfit_packer extension."""
        ext_dir = Path(__file__).parent
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        so_path = ext_dir / f"bestfit_packer{ext_suffix}"

        # Always run make — it checks timestamps and only recompiles when
        # bestfit_packer.cpp is newer than the .so (no-op if up to date).
        result = subprocess.run(
            ["make", "-C", str(ext_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to compile bestfit_packer:\n{result.stderr}"
            )
        if "Nothing to be done" not in result.stdout and so_path.exists():
            logger.info("bestfit_packer compiled successfully")

        spec = importlib.util.spec_from_file_location(
            "bestfit_packer", str(so_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Bump when the on-disk cache layout changes (e.g. swap from .npz to
    # mmap-backed binary files, change ref dtype, ...). Hashed into the cache
    # key so old caches auto-invalidate on format change. Cosmetic edits to
    # bestfit_packer.cpp no longer invalidate caches — only intentional bumps
    # of this constant do.
    _CACHE_FORMAT_VERSION = "v2"

    def _compute_bestfit_cache_key(self, epoch: int) -> str:
        """Compute MD5 hash for cache key based on all deterministic params."""
        hasher = hashlib.md5()
        hasher.update(f"fmt={self._CACHE_FORMAT_VERSION}".encode())
        # Include the absolute chunks_dir path so two different datasets that
        # happen to share chunk filenames (e.g. chunk_0001) but live under
        # different roots don't collide when they share a cache_dir.
        hasher.update(f"cd={str(self.chunks_dir.resolve())}".encode())
        hasher.update(f"seed={self.seed + epoch}".encode())
        hasher.update(f"sl={self.seq_len}".encode())
        hasher.update(f"ml={self.min_sequence_length}".encode())
        hasher.update(f"eos={self.eos_id}".encode())
        hasher.update(f"bs={self.best_fit_buffer_size}".encode())
        # Split params also matter — training (exclude_first_n) and validation
        # (use_only_first_n) views of the same chunks must hash differently.
        hasher.update(f"exf={self.exclude_first_n_per_chunk}".encode())
        hasher.update(f"unf={self.use_only_first_n_per_chunk}".encode())
        # Include chunk info for cache invalidation. Per-chunk inline-EOS
        # state is included so a re-tokenized corpus that suddenly has EOS
        # (or loses it) invalidates the cache rather than returning a stale
        # plan whose fragment budgets were sized for a different EOS regime.
        for chunk_id in self.chunk_order:
            n_docs = len(self._all_chunk_doc_lengths[chunk_id])
            chunk_name = Path(self.all_chunks[chunk_id]).name
            inline_eos = self._chunk_has_inline_eos[chunk_id]
            hasher.update(f"{chunk_name}:{n_docs}:eos_inline={inline_eos}".encode())
        return hasher.hexdigest()[:16]

    def _resolve_cache_dir(self) -> Path:
        """Pick the directory to read/write packing-plan files in.

        Honors `cache_dir` constructor arg when set; otherwise falls back to
        `<chunks_dir>/.packing_cache` (legacy behavior).
        """
        if self.cache_dir is not None:
            return self.cache_dir
        return self.chunks_dir / ".packing_cache"

    def _cache_paths(self, epoch: int) -> Dict[str, Path]:
        """Return the four files that make up one cached plan."""
        cache_dir = self._resolve_cache_dir()
        cache_key = self._compute_bestfit_cache_key(epoch)
        prefix = cache_dir / f"bestfit_{cache_key}"
        return {
            "counts": prefix.with_suffix(".counts.bin"),
            "refs": prefix.with_suffix(".refs.bin"),
            "offsets": prefix.with_suffix(".offsets.bin"),
            "manifest": prefix.with_suffix(".manifest.json"),
        }

    def _build_or_load_bestfit_plan(self, epoch: int) -> None:
        """Build or load cached best-fit packing plan for the given epoch.

        On-disk layout (one cache slot, all four files share one cache_key):
          bestfit_<key>.counts.bin    int32   [num_sequences]
          bestfit_<key>.refs.bin      int64   [num_refs, 4]   (chunk_order_idx,
                                                               eff_doc_idx,
                                                               token_start,
                                                               token_end)
          bestfit_<key>.offsets.bin   int64   [num_sequences + 1] (cumsum)
          bestfit_<key>.manifest.json {format_version, num_sequences, num_refs}

        The arrays are np.memmap'd on read, so per-rank resident memory tracks
        only the pages actively touched by the rank's sequence stride —
        independent of total dataset size. Cold start on a multi-T-token
        dataset still reads the manifest + headers; refs.bin pages in lazily.
        """
        paths = self._cache_paths(epoch)
        manifest_path = paths["manifest"]

        if manifest_path.exists():
            try:
                self._load_cached_plan(paths)
                logger.info(
                    f"Loaded best-fit plan from cache: "
                    f"{paths['counts'].with_suffix('').name} "
                    f"({self.total_sequences} sequences, "
                    f"{len(self._bestfit_doc_refs)} fragments)"
                )
                return
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Cache present but unreadable ({e!r}); rebuilding."
                )

        # GLOBAL build lock: the plan build flattens all doc-lengths + runs the
        # C++ packer (~several GB peak). With N dataloader workers per node ALL
        # building concurrently the first time, that is Nx the memory -> host OOM
        # on big stores (e.g. the 300B distill store, 326M docs, 4 workers/node).
        # Serialize via an atomic lockfile: exactly ONE process builds; the rest
        # wait then load the cache. Stale lock (builder crash) reclaimed after 1h.
        import os as _os, time as _time
        lock_path = manifest_path.with_name(manifest_path.name + ".building.lock")
        while not manifest_path.exists():
            try:
                _fd = _os.open(str(lock_path), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
                _os.write(_fd, str(_os.getpid()).encode()); _os.close(_fd)
            except FileExistsError:
                try:
                    _stale = (_time.time() - _os.path.getmtime(lock_path)) > 3600
                except OSError:
                    _stale = False
                if _stale:
                    try: _os.unlink(lock_path)
                    except OSError: pass
                else:
                    _time.sleep(5)
                continue
            try:
                if not manifest_path.exists():
                    logger.info("Building best-fit plan (this process holds the build lock) ...")
                    self._build_plan_streaming(epoch, paths)
            finally:
                try: _os.unlink(lock_path)
                except OSError: pass
        self._load_cached_plan(paths)

    def _build_plan_streaming(self, epoch: int, paths: Dict[str, Path]) -> None:
        """Run the C++ packer with file output, then materialize offsets."""
        cache_dir = paths["counts"].parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building best-fit packing plan via C++ extension...")
        t0 = time.monotonic()
        packer = self._compile_bestfit_packer()

        # Prepare inputs: flatten all effective doc lengths in permuted chunk order
        all_doc_lengths = []
        all_chunk_ids = []
        all_doc_indices = []
        for order_idx, chunk_id in enumerate(self.chunk_order):
            lengths = self._all_chunk_doc_lengths[chunk_id]
            n_docs = len(lengths)
            all_doc_lengths.append(lengths)
            all_chunk_ids.append(np.full(n_docs, order_idx, dtype=np.int32))
            all_doc_indices.append(np.arange(n_docs, dtype=np.int32))

        doc_lengths = np.concatenate(all_doc_lengths).astype(np.int64)
        chunk_ids = np.concatenate(all_chunk_ids)
        doc_indices = np.concatenate(all_doc_indices)
        total_tokens = int(doc_lengths.sum())

        # Free the per-chunk python-side lists ASAP — for big datasets these
        # already weigh several GB and we don't need them during the build.
        del all_doc_lengths, all_chunk_ids, all_doc_indices

        # Stage every file under <name>.tmp so a crashed build never leaves a
        # half-written cache that subsequent runs would mistake for valid.
        tmp_counts = paths["counts"].with_suffix(paths["counts"].suffix + ".tmp")
        tmp_refs = paths["refs"].with_suffix(paths["refs"].suffix + ".tmp")
        tmp_offsets = paths["offsets"].with_suffix(paths["offsets"].suffix + ".tmp")
        tmp_manifest = paths["manifest"].with_suffix(paths["manifest"].suffix + ".tmp")
        # Clean any stragglers from a previous failed run.
        for p in (tmp_counts, tmp_refs, tmp_offsets, tmp_manifest):
            if p.exists():
                p.unlink()

        try:
            num_sequences, num_refs = packer.build_bestfit_plan(
                doc_lengths, chunk_ids, doc_indices,
                self.tokens_per_seq,
                self.min_sequence_length,
                self.best_fit_buffer_size,
                self.seed + epoch,
                str(tmp_counts),
                str(tmp_refs),
            )
        except Exception:
            for p in (tmp_counts, tmp_refs):
                if p.exists():
                    p.unlink()
            raise

        elapsed = time.monotonic() - t0
        logger.info(
            f"Best-fit plan built in {elapsed:.2f}s: "
            f"{num_sequences} sequences from {len(doc_lengths)} docs "
            f"({total_tokens / 1e6:.1f}M tokens, {num_refs} fragments)"
        )

        # Use the actual file size on disk as the source of truth for the count.
        # The C++ packer's returned num_sequences has been observed to mismatch
        # the bytes actually persisted to tmp_counts (e.g. epoch-1 build with
        # 1.8 T tokens: packer reported 439 371 959 but the mmap fails with
        # "mmap length is greater than file size"). Until that's root-caused
        # in the C++ packer, prefer st_size//4 — it is consistent with what
        # the offsets cumsum will see in the next step, and an off-by-a-few
        # sequences in the plan is harmless at this scale.
        counts_bytes = tmp_counts.stat().st_size
        actual_num_sequences = counts_bytes // 4
        if counts_bytes % 4 != 0:
            raise RuntimeError(
                f"{tmp_counts} size ({counts_bytes}) is not a multiple of 4 — "
                f"counts file is corrupted"
            )
        if actual_num_sequences != num_sequences:
            logger.warning(
                "Packer reported num_sequences=%d but tmp_counts holds %d "
                "(delta %+d). Trusting the file size.",
                num_sequences, actual_num_sequences,
                actual_num_sequences - num_sequences,
            )
            num_sequences = actual_num_sequences

        # Similarly trust the refs file size: each ref is 4 × int64 = 32 bytes.
        refs_bytes = tmp_refs.stat().st_size
        if refs_bytes % 32 != 0:
            raise RuntimeError(
                f"{tmp_refs} size ({refs_bytes}) is not a multiple of 32 — "
                f"refs file is corrupted"
            )
        actual_num_refs = refs_bytes // 32
        if actual_num_refs != num_refs:
            logger.warning(
                "Packer reported num_refs=%d but tmp_refs holds %d (delta %+d). "
                "Trusting the file size.",
                num_refs, actual_num_refs, actual_num_refs - num_refs,
            )
            num_refs = actual_num_refs

        # Materialize the offsets file via mmap-backed cumsum. Keeps peak RAM
        # bounded — only the cumsum's working state is in memory, the input
        # and output are pages on disk that the OS swaps as needed.
        counts_view = np.memmap(
            tmp_counts, dtype=np.int32, mode="r", shape=(num_sequences,)
        )
        offsets_view = np.memmap(
            tmp_offsets, dtype=np.int64, mode="w+",
            shape=(num_sequences + 1,),
        )
        offsets_view[0] = 0
        if num_sequences > 0:
            np.cumsum(counts_view, dtype=np.int64, out=offsets_view[1:])
        offsets_view.flush()
        del counts_view, offsets_view

        # Manifest is the "completion marker" — written last, renamed last.
        manifest = {
            "format_version": self._CACHE_FORMAT_VERSION,
            "num_sequences": int(num_sequences),
            "num_refs": int(num_refs),
        }
        with open(tmp_manifest, "w") as fh:
            json.dump(manifest, fh)

        # Atomic-ish: rename data files first, then manifest. A reader that
        # checks manifest.exists() will only succeed if all three data files
        # were already in place.
        tmp_counts.replace(paths["counts"])
        tmp_refs.replace(paths["refs"])
        tmp_offsets.replace(paths["offsets"])
        tmp_manifest.replace(paths["manifest"])

        logger.info(f"Best-fit plan cached to {paths['manifest'].parent}")

    def _load_cached_plan(self, paths: Dict[str, Path]) -> None:
        """Open the four cache files via mmap and wire them into self.*."""
        with open(paths["manifest"]) as fh:
            manifest = json.load(fh)

        if manifest.get("format_version") != self._CACHE_FORMAT_VERSION:
            raise ValueError(
                f"cache format mismatch: manifest says "
                f"{manifest.get('format_version')!r}, expected "
                f"{self._CACHE_FORMAT_VERSION!r}"
            )

        num_sequences = int(manifest["num_sequences"])
        num_refs = int(manifest["num_refs"])

        # mmap mode 'r' is read-only and shared — pages are demand-faulted by
        # the OS and reclaimed under memory pressure. The numpy interface
        # (indexing, slicing) is identical to a regular ndarray; downstream
        # code in _read_packed_sequence works unchanged.
        if num_sequences == 0:
            self._bestfit_seq_doc_counts = np.zeros((0,), dtype=np.int32)
            self._bestfit_seq_offsets = np.zeros((1,), dtype=np.int64)
        else:
            self._bestfit_seq_doc_counts = np.memmap(
                paths["counts"], dtype=np.int32, mode="r",
                shape=(num_sequences,),
            )
            self._bestfit_seq_offsets = np.memmap(
                paths["offsets"], dtype=np.int64, mode="r",
                shape=(num_sequences + 1,),
            )

        if num_refs == 0:
            self._bestfit_doc_refs = np.zeros((0, 4), dtype=np.int64)
        else:
            self._bestfit_doc_refs = np.memmap(
                paths["refs"], dtype=np.int64, mode="r",
                shape=(num_refs, 4),
            )

        # Override total_sequences and steps_per_epoch from plan
        self.total_sequences = num_sequences
        self.steps_per_epoch = self.total_sequences // self.global_batch_size

    # ── Read a packed sequence from the plan ─────────────────────────────

    def _read_packed_sequence(
        self, global_seq_id: int
    ) -> Dict[str, Any]:
        """Read a packed sequence from the best-fit plan.

        Looks up the pre-computed plan to find which document fragments
        compose this sequence, then reads their tokens from chunk files.
        """
        offset = int(self._bestfit_seq_offsets[global_seq_id])
        count = int(self._bestfit_seq_doc_counts[global_seq_id])
        refs = self._bestfit_doc_refs[offset : offset + count]

        all_tokens = []
        seqlens = []

        for ref in refs:
            chunk_order_idx = int(ref[0])
            eff_doc_idx = int(ref[1])
            token_start = int(ref[2])
            token_end = int(ref[3])

            chunk_id = self.chunk_order[chunk_order_idx]
            chunk_data = self._get_chunk_data(chunk_id)

            raw_doc_idx = chunk_data.effective_to_raw_idx[eff_doc_idx]
            raw_pointer, raw_length, _ = chunk_data.index_reader[raw_doc_idx]
            dtype = chunk_data.index_reader.dtype
            dtype_size = np.dtype(dtype).itemsize

            frag_len = token_end - token_start
            chunk_inline_eos = self._chunk_has_inline_eos[chunk_id]

            if self.has_eos and not chunk_inline_eos:
                # Fragment may span raw tokens + EOS at position raw_length
                raw_start = min(token_start, raw_length)
                raw_end = min(token_end, raw_length)
                raw_count = raw_end - raw_start

                if raw_count > 0:
                    read_offset = raw_pointer + raw_start * dtype_size
                    tokens = chunk_data.bin_reader.read(
                        dtype=dtype, count=raw_count, offset=read_offset,
                    )
                    all_tokens.append(tokens)

                if token_end > raw_length:
                    # Defensive dedup: if the just-read fragment already ends
                    # with eos_id (mixed-EOS chunk that probed as no-inline),
                    # skip the synthetic append. _read_all_chunk_doc_lengths
                    # logged a warning when this can happen.
                    last_token = (
                        int(tokens[-1]) if raw_count > 0 else None
                    )
                    if last_token != int(self.eos_id):
                        all_tokens.append(
                            np.array([self.eos_id], dtype=dtype)
                        )
            else:
                read_offset = raw_pointer + token_start * dtype_size
                tokens = chunk_data.bin_reader.read(
                    dtype=dtype, count=frag_len, offset=read_offset,
                )
                all_tokens.append(tokens)

            seqlens.append(frag_len)

        if all_tokens:
            sequence = np.concatenate(all_tokens)
        else:
            sequence = np.array([], dtype=np.int64)

        return {
            'tokens': [sequence],
            'seqlen': seqlens,
        }

    # ── Iterator protocol ────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        # Strided batch assignment — same logic as DeterministicPackedDataset.
        # Step S batch = {S + k*E : k = 0..GBS-1}, dp_world_size independent.
        local_offset = self.global_sequence_id % self.sequences_per_rank
        step = self.global_sequence_id // self.sequences_per_rank

        if local_offset == 0 and step >= self.steps_per_epoch:
            if self.infinite:
                self.epoch_counter += 1
                self._setup_global_ordering(self.epoch_counter)
                self._build_or_load_bestfit_plan(self.epoch_counter)
                self.global_sequence_id = 0
                step = 0
                local_offset = 0
                logger.info(
                    f"Rank {self.dp_rank}: starting epoch {self.epoch_counter}"
                )
            else:
                raise StopIteration

        batch_position = self.dp_rank * self.sequences_per_rank + local_offset
        global_seq = step + batch_position * self.steps_per_epoch

        # Read the packed sequence from the plan
        sample = self._read_packed_sequence(global_seq)

        self.global_sequence_id += 1
        self.sample_counter += 1

        return sample

    def __len__(self) -> int:
        """Total packed sequences this rank will yield per epoch."""
        return self.steps_per_epoch * self.sequences_per_rank

    # ── Checkpointing ────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Minimal training-checkpoint state. Mirrors DPD's protocol so the
        same trainer code can restore either dataloader.
        """
        return {
            "global_sequence_id": int(self.global_sequence_id),
            "epoch_counter": int(self.epoch_counter),
            "sample_counter": int(self.sample_counter),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore from `state_dict()`. Re-runs ordering + plan build for the
        saved epoch so the next `__next__` resumes at the saved sequence."""
        self.epoch_counter = int(state_dict["epoch_counter"])
        self.global_sequence_id = int(state_dict["global_sequence_id"])
        self.sample_counter = int(state_dict.get("sample_counter", 0))
        self._setup_global_ordering(self.epoch_counter)
        self._build_or_load_bestfit_plan(self.epoch_counter)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle state without file handles."""
        state = self.__dict__.copy()
        # Remove loaded chunk data (has file handles)
        state.pop('_loaded_chunks', None)
        # Remove best-fit plan arrays (reloaded from cache on restore)
        state.pop('_bestfit_seq_doc_counts', None)
        state.pop('_bestfit_doc_refs', None)
        state.pop('_bestfit_seq_offsets', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and reinitialize."""
        self.__dict__.update(state)
        self._loaded_chunks = {}
        # Reload best-fit plan from cache
        self._bestfit_seq_doc_counts = None
        self._bestfit_doc_refs = None
        self._bestfit_seq_offsets = None
        self._build_or_load_bestfit_plan(self.epoch_counter)

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
            'packing_mode': 'best_fit',
            'best_fit_buffer_size': self.best_fit_buffer_size,
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
