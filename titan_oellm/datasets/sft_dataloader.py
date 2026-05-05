"""
SFT DataLoader for instruction-formatted and reasoning datasets.

Supports:
- HuggingFace datasets (any dataset with text/conversation data)
- Local JSONL files
- Instruction formats: Alpaca, ChatML, ShareGPT, etc.
- Reasoning formats: Chain-of-Thought, step-by-step reasoning
- Auto-detection of dataset structure

The dataloader formats the data and creates loss masks to only compute
loss on the response tokens, not the instruction tokens.
"""

import json
import logging
import bisect
import time
import os
import importlib
from pathlib import Path
from typing import Any, Optional, Sequence, Union, Sized, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict, concatenate_datasets, load_from_disk
from titan_oellm.datasets.dataloader.mmap_dataset_chunked import _IndexReader, _RandomAccessBinReader

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig

from titan_oellm.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def _maybe_tqdm(iterable, total: int, desc: str):
    """Return a tqdm iterator when available, otherwise return iterable unchanged."""
    try:
        tqdm_mod = importlib.import_module("tqdm.auto")
        rank = int(os.environ.get("RANK", "0"))
        disable = rank != 0
        return tqdm_mod.tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
            disable=disable,
        )
    except Exception:
        return iterable


class _ChunkedArrayReader:
    """Read fixed-length arrays from chunk_*.idx/bin files."""

    def __init__(self, chunks_dir: Union[str, Path]):
        self.chunks_dir = Path(chunks_dir)
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunk directory not found: {self.chunks_dir}")

        idx_files = sorted(self.chunks_dir.glob("chunk_*.idx"))
        if not idx_files:
            raise RuntimeError(f"No chunk index files found under {self.chunks_dir}")

        self.idx_readers = []
        self.bin_readers = []
        self.chunk_starts = []
        self.chunk_lengths = []
        total = 0
        for idx_file in idx_files:
            prefix = str(idx_file).replace(".idx", "")
            bin_file = Path(f"{prefix}.bin")
            if not bin_file.exists():
                raise RuntimeError(f"Missing bin file for chunk index: {idx_file}")

            idx_reader = _IndexReader(str(idx_file))
            bin_reader = _RandomAccessBinReader(str(bin_file))
            chunk_len = len(idx_reader)

            self.idx_readers.append(idx_reader)
            self.bin_readers.append(bin_reader)
            self.chunk_starts.append(total)
            self.chunk_lengths.append(chunk_len)
            total += chunk_len

        self.total_rows = total
        if self.total_rows == 0:
            raise RuntimeError(f"No rows found in chunked dataset: {self.chunks_dir}")

    def __len__(self) -> int:
        return self.total_rows

    def read(self, global_idx: int) -> np.ndarray:
        if global_idx < 0 or global_idx >= self.total_rows:
            raise IndexError(global_idx)

        chunk_idx = bisect.bisect_right(self.chunk_starts, global_idx) - 1
        local_idx = global_idx - self.chunk_starts[chunk_idx]
        pointer, length, _ = self.idx_readers[chunk_idx][local_idx]
        return self.bin_readers[chunk_idx].read(
            dtype=self.idx_readers[chunk_idx].dtype,
            count=length,
            offset=pointer,
        )


class PrepackedInstructionDataset(Dataset):
    """Dataset wrapper for offline prepacked SFT rows saved with dataset_packer.py."""

    def __init__(self, data_path: str):
        p = Path(data_path)
        self._mode = "hf"

        input_npy = p / "input.npy"
        labels_npy = p / "labels.npy"
        mask_npy = p / "loss_mask.npy"
        if input_npy.exists() and labels_npy.exists() and mask_npy.exists():
            self._mode = "npy"
            self.input_arr = np.load(str(input_npy), mmap_mode="r")
            self.labels_arr = np.load(str(labels_npy), mmap_mode="r")
            self.mask_arr = np.load(str(mask_npy), mmap_mode="r")
            if self.input_arr.shape != self.labels_arr.shape or self.input_arr.shape != self.mask_arr.shape:
                raise RuntimeError(
                    f"Prepacked .npy arrays must have matching shapes, got "
                    f"input={self.input_arr.shape}, labels={self.labels_arr.shape}, loss_mask={self.mask_arr.shape}"
                )
        elif (p / "input").exists() and (p / "labels").exists() and (p / "loss_mask").exists():
            self._mode = "chunked"
            self.input_reader = _ChunkedArrayReader(p / "input")
            self.labels_reader = _ChunkedArrayReader(p / "labels")
            self.mask_reader = _ChunkedArrayReader(p / "loss_mask")
            if len(self.input_reader) != len(self.labels_reader) or len(self.input_reader) != len(self.mask_reader):
                raise RuntimeError(
                    "Chunked prepacked input/labels/loss_mask row counts do not match: "
                    f"input={len(self.input_reader)}, labels={len(self.labels_reader)}, loss_mask={len(self.mask_reader)}"
                )
        else:
            self.data = load_from_disk(data_path)
            required = {"input", "labels", "loss_mask"}
            missing = required - set(self.data.column_names)
            if missing:
                raise RuntimeError(
                    f"Prepacked dataset at {data_path} missing columns: {sorted(missing)}"
                )

    def __len__(self) -> int:
        if self._mode == "npy":
            return int(self.input_arr.shape[0])
        if self._mode == "chunked":
            return len(self.input_reader)
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._mode == "npy":
            return {
                "input": torch.tensor(self.input_arr[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels_arr[idx], dtype=torch.long),
                "loss_mask": torch.tensor(self.mask_arr[idx], dtype=torch.long),
            }

        if self._mode == "chunked":
            return {
                "input": torch.tensor(self.input_reader.read(idx), dtype=torch.long),
                "labels": torch.tensor(self.labels_reader.read(idx), dtype=torch.long),
                "loss_mask": torch.tensor(self.mask_reader.read(idx), dtype=torch.long),
            }

        row = self.data[idx]
        return {
            "input": torch.tensor(row["input"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.long),
        }


def _collate_pad(batch: list[dict[str, torch.Tensor]], pad_id: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Pad variable-length SFT samples in a batch to the max sample length."""
    max_len = max(item["input"].numel() for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    loss_masks = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input"].numel()
        input_ids[i, :length] = item["input"]
        labels[i, :length] = item["labels"]
        loss_masks[i, :length] = item["loss_mask"]

    return {
        "input": input_ids,
        "loss_mask": loss_masks,
    }, labels


def _collate_prepacked(batch: list[dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Stack fixed-length prepacked rows without online packing/padding logic."""
    input_ids = torch.stack([item["input"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    loss_masks = torch.stack([item["loss_mask"] for item in batch], dim=0)
    return {"input": input_ids, "loss_mask": loss_masks}, labels


class GlobalOBFDPackedDataset(Dataset):
    """Global best-fit-decreasing packing over the entire tokenized SFT dataset."""

    def __init__(self, base_dataset: Dataset, seq_len: int, pad_id: int):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.pad_id = pad_id

        build_start = time.time()
        lengths = []
        base_len = len(cast(Sized, self.base_dataset))
        logger.info(
            "[online-obfd] building global packing plan: samples=%d, seq_len=%d",
            base_len,
            seq_len,
        )

        scan_start = time.time()
        for i in _maybe_tqdm(range(base_len), total=base_len, desc="online-obfd: scan lengths"):
            item = base_dataset[i]
            lengths.append(min(int(item["input"].numel()), seq_len))
        logger.info(
            "[online-obfd] length scan completed in %.2fs",
            time.time() - scan_start,
        )

        sort_start = time.time()
        sorted_base_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        sorted_lengths = [lengths[i] for i in sorted_base_indices]
        logger.info(
            "[online-obfd] sorting completed in %.2fs",
            time.time() - sort_start,
        )

        bin_items: list[list[int]] = []
        bin_used: list[int] = []
        pack_start = time.time()
        total_docs = len(sorted_lengths)
        packed_token_total = 0
        for sorted_doc_id in _maybe_tqdm(range(total_docs), total=total_docs, desc="online-obfd: assign bins"):
            doc_len = sorted_lengths[sorted_doc_id]
            if doc_len <= 0:
                continue

            packed_token_total += doc_len

            best_idx = -1
            best_remaining = None
            for i, used in enumerate(bin_used):
                remaining = seq_len - used
                if doc_len <= remaining:
                    candidate_remaining = remaining - doc_len
                    if best_remaining is None or candidate_remaining < best_remaining:
                        best_remaining = candidate_remaining
                        best_idx = i

            if best_idx >= 0:
                bin_items[best_idx].append(sorted_doc_id)
                bin_used[best_idx] += doc_len
            else:
                bin_items.append([sorted_doc_id])
                bin_used.append(doc_len)

        self.sorted_base_indices = sorted_base_indices
        self.bins = bin_items

        build_secs = time.time() - build_start
        pack_secs = time.time() - pack_start
        total_capacity = max(1, len(self.bins) * self.seq_len)
        fill_ratio = float(packed_token_total) / float(total_capacity)
        logger.info(
            "[online-obfd] packing complete: bins=%d, fill_ratio=%.4f, pack_time=%.2fs, total_build_time=%.2fs",
            len(self.bins),
            fill_ratio,
            pack_secs,
            build_secs,
        )
    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        packed_input = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        packed_labels = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        packed_mask = torch.zeros((self.seq_len,), dtype=torch.long)

        pos = 0
        for sorted_doc_id in self.bins[idx]:
            base_idx = self.sorted_base_indices[sorted_doc_id]
            sample = self.base_dataset[base_idx]

            in_t = sample["input"][: self.seq_len - pos]
            lab_t = sample["labels"][: self.seq_len - pos]
            mask_t = sample["loss_mask"][: self.seq_len - pos]
            length = int(in_t.numel())
            if length <= 0:
                continue

            packed_input[pos:pos + length] = in_t
            packed_labels[pos:pos + length] = lab_t
            packed_mask[pos:pos + length] = mask_t
            pos += length
            if pos >= self.seq_len:
                break

        return {
            "input": packed_input,
            "labels": packed_labels,
            "loss_mask": packed_mask,
        }


class SequentialPackedSFTDataset(Dataset):
    """OLMo3-style sequential (greedy) document packing for SFT data.

    Documents from the base InstructionDataset are concatenated in order into
    fixed-length sequences. Documents that exceed the boundary are split across
    consecutive sequences (identical to DeterministicPackedDataset for pretraining,
    but operating on SFT data with loss masks instead of raw token streams).

    Document boundaries are tracked via a 'seqlen' list so the collator can
    produce intra-document attention masks (block-causal / cu_seqlens).

    Returns per item:
        {'input': [S], 'labels': [S], 'loss_mask': [S], 'seqlen': [int, ...]}
    where seqlen lists the length of each document chunk packed into this sequence.
    """

    def __init__(self, base_dataset: Dataset, seq_len: int, pad_id: int, index_cache_path: Optional[str] = None):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.pad_id = pad_id

        import torch.distributed as dist
        global_rank = int(os.environ.get("RANK", "0"))
        is_distributed = dist.is_initialized()

        raw_lengths = self._load_or_build_index(
            base_dataset, seq_len, index_cache_path, global_rank, is_distributed
        )

        # Keep only non-empty documents; store original indices for __getitem__.
        self._doc_base_idx = [i for i, l in enumerate(raw_lengths) if l > 0]
        self._doc_lengths  = np.array([raw_lengths[i] for i in self._doc_base_idx], dtype=np.int64)
        self._doc_cumsum   = np.concatenate([[0], np.cumsum(self._doc_lengths)])
        self._total_tokens = int(self._doc_cumsum[-1])

        # Ceil-divide: the last sequence may be shorter than seq_len (padded).
        self._n_seqs = max(1, (self._total_tokens + seq_len - 1) // seq_len)

        # Precompute per-sequence document boundary lists (cheap, just integers).
        self._seq_seqlens: list[list[int]] = [
            self._chunks_in_range(i * seq_len, min((i + 1) * seq_len, self._total_tokens))
            for i in range(self._n_seqs)
        ]

        logger.info(
            "[sequential-pack] %d docs → %d sequences "
            "(seq_len=%d, total_tokens=%d, avg_docs_per_seq=%.2f)",
            len(self._doc_base_idx),
            self._n_seqs,
            seq_len,
            self._total_tokens,
            sum(len(s) for s in self._seq_seqlens) / max(1, self._n_seqs),
        )

    def _load_or_build_index(
        self,
        base_dataset: Dataset,
        seq_len: int,
        index_cache_path: Optional[str],
        global_rank: int,
        is_distributed: bool,
    ) -> list[int]:
        cache_file = Path(index_cache_path) if index_cache_path else None

        if cache_file and cache_file.exists():
            logger.info("[sequential-pack] rank %d loading index from cache: %s", global_rank, cache_file)
            return np.load(str(cache_file)).tolist()

        if cache_file and not cache_file.exists():
            raise RuntimeError(
                f"sequential_index_cache_path is set but the file does not exist: {cache_file}\n"
                f"Run build_sequential_index.py first to pre-build the index."
            )

        # No cache configured — all ranks scan independently (slow on large datasets).
        base_len = len(cast(Sized, base_dataset))
        logger.warning(
            "[sequential-pack] no index cache configured; rank %d scanning %d documents. "
            "Set data.sequential_index_cache_path and run build_sequential_index.py to avoid this.",
            global_rank, base_len,
        )
        raw_lengths = []
        for i in _maybe_tqdm(range(base_len), total=base_len, desc="sequential-pack: scan"):
            item = base_dataset[i]
            raw_lengths.append(int(item["input"].numel()))
        return raw_lengths

    # ------------------------------------------------------------------
    def _chunks_in_range(self, seq_start: int, seq_end: int) -> list[int]:
        """Document-chunk lengths that fall within token range [seq_start, seq_end)."""
        seqlens: list[int] = []
        first_doc = int(np.searchsorted(self._doc_cumsum[1:], seq_start, side="right"))
        doc_idx = first_doc
        while doc_idx < len(self._doc_lengths):
            doc_start = int(self._doc_cumsum[doc_idx])
            doc_end   = int(self._doc_cumsum[doc_idx + 1])
            if doc_start >= seq_end:
                break
            chunk_len = min(seq_end, doc_end) - max(seq_start, doc_start)
            if chunk_len > 0:
                seqlens.append(chunk_len)
            doc_idx += 1
        return seqlens

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n_seqs

    def __getitem__(self, seq_idx: int) -> dict:
        seq_start  = seq_idx * self.seq_len
        seq_end    = min(seq_start + self.seq_len, self._total_tokens)
        actual_len = seq_end - seq_start

        inp = torch.zeros(self.seq_len, dtype=torch.long)
        lab = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        msk = torch.zeros(self.seq_len, dtype=torch.long)

        first_doc = int(np.searchsorted(self._doc_cumsum[1:], seq_start, side="right"))
        write_pos = 0
        doc_idx   = first_doc

        while write_pos < actual_len and doc_idx < len(self._doc_lengths):
            doc_start = int(self._doc_cumsum[doc_idx])
            doc_end   = int(self._doc_cumsum[doc_idx + 1])
            if doc_start >= seq_end:
                break

            offset_in_doc = max(seq_start, doc_start) - doc_start
            chunk_len     = min(seq_end, doc_end) - (doc_start + offset_in_doc)

            if chunk_len > 0:
                item = self.base_dataset[self._doc_base_idx[doc_idx]]
                sl   = slice(offset_in_doc, offset_in_doc + chunk_len)
                inp[write_pos : write_pos + chunk_len] = item["input"][sl]
                lab[write_pos : write_pos + chunk_len] = item["labels"][sl]
                msk[write_pos : write_pos + chunk_len] = item["loss_mask"][sl]
                write_pos += chunk_len

            doc_idx += 1

        return {
            "input":    inp,
            "labels":   lab,
            "loss_mask": msk,
            "seqlen":   self._seq_seqlens[seq_idx],
        }


def _collate_sft_doc_mask(
    batch: list[dict],
    seq_len: int,
    pad_id: int = 0,
    use_flash_attention: bool = False,
    max_cu_seqlens_size: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate sequentially-packed SFT samples with intra-document attention masking.

    Reads the 'seqlen' boundary list from each sample and produces:
    - Flash Attention path: cu_seqlens over the flattened [B*S] space.
      A padding segment is appended per item so consecutive items start at the
      correct flat offset (prevents cross-item attention bleed).
    - SDPA path: attention_masks [B, 1, S, S] = causal ∧ same-document mask.

    Returns:
        input_dict: {'input', 'loss_mask', 'cu_seqlens'/'attention_masks'}
        labels:     [B, S]  (raw; SFTTrainer applies loss_mask to get ignore_index)
    """
    input_ids   = torch.stack([item["input"]    for item in batch], dim=0)
    labels      = torch.stack([item["labels"]   for item in batch], dim=0)
    loss_masks  = torch.stack([item["loss_mask"] for item in batch], dim=0)
    sample_seqlens = [item["seqlen"] for item in batch]

    if use_flash_attention:
        # Each batch item i occupies flat positions [i*seq_len, (i+1)*seq_len).
        # Real doc chunks fill the front; the remainder is padding.
        # We emit a padding segment at the end of each item so the next item's
        # first doc starts at the right flat offset.
        flat_seqlens: list[int] = []
        for seqlens in sample_seqlens:
            flat_seqlens.extend(seqlens)
            pad_len = seq_len - sum(seqlens)
            if pad_len > 0:
                flat_seqlens.append(pad_len)

        flat_t     = torch.tensor(flat_seqlens, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(flat_t, dim=0, dtype=torch.int32), (1, 0))

        if max_cu_seqlens_size is not None:
            if cu_seqlens.shape[0] < max_cu_seqlens_size:
                cu_seqlens = F.pad(
                    cu_seqlens,
                    (0, max_cu_seqlens_size - cu_seqlens.shape[0]),
                    value=cu_seqlens[-1].item(),
                )
            elif cu_seqlens.shape[0] > max_cu_seqlens_size:
                # Documents shorter than min_doc_len produce more segments than
                # budgeted. Truncate to the budget; the last entry must equal the
                # total token count so flash_attn sees the full sequence.
                cu_seqlens = torch.cat([cu_seqlens[:max_cu_seqlens_size - 1], cu_seqlens[-1:]])

        return {
            "input":      input_ids,
            "loss_mask":  loss_masks,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": seq_len,
        }, labels

    else:
        # SDPA: build [B, 1, S, S] boolean block-causal mask.
        # Combines a lower-triangular causal mask with a same-document mask so
        # tokens cannot attend across document boundaries.
        padded_S = input_ids.shape[1]
        causal   = torch.ones(padded_S, padded_S, dtype=torch.bool).tril()
        doc_masks = []
        for seqlens in sample_seqlens:
            doc_ids = torch.zeros(padded_S, dtype=torch.int32)
            pos = 0
            for doc_idx, doc_len in enumerate(seqlens):
                end = min(pos + doc_len, padded_S)
                doc_ids[pos:end] = doc_idx
                pos = end
            # Padding positions keep doc_id=0; causal masking already prevents
            # real tokens from attending to padding (it sits in their future).
            doc_masks.append(doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))
        doc_mask = torch.stack(doc_masks).unsqueeze(1)   # [B, 1, S, S]

        return {
            "input":           input_ids,
            "loss_mask":       loss_masks,
            "attention_masks": causal.unsqueeze(0).unsqueeze(0) & doc_mask,
        }, labels


def detect_format(example: dict) -> str:
    """Auto-detect the instruction format from a dataset example."""
    # Check for reasoning formats first (more specific)
    if "steps" in example or "reasoning" in example:
        return "reasoning_steps"
    elif ("chain_of_thought" in example or "cot" in example or 
          "thought" in example or "thinking" in example):
        return "cot"
    elif "problem" in example and ("solution" in example or "answer" in example):
        return "problem_solution"
    
    # Check for conversation/instruction formats
    if "messages" in example:
        return "chatml"
    elif "conversations" in example:
        return "sharegpt"
    elif "instruction" in example and "output" in example:
        return "alpaca"
    elif "prompt" in example and "completion" in example:
        return "prompt_completion"
    elif "text" in example:
        return "text"
    else:
        # Try to find any text-like fields
        for key in ["question", "query", "input"]:
            if key in example:
                return "auto"
        return "unknown"


class InstructionDataset(Dataset):
    """
    Dataset for instruction-formatted data used in supervised fine-tuning.
    
    Supports:
    - HuggingFace datasets (load via dataset name/path)
    - Local JSONL files
    - Auto-detection of format
    - Multiple instruction formats
    """

    @staticmethod
    def _hf_cache_roots() -> list[Path]:
        roots: list[Path] = []
        hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
        if hf_datasets_cache:
            roots.append(Path(hf_datasets_cache))

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "datasets")

        roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

        seen: set[Path] = set()
        existing_roots: list[Path] = []
        for root in roots:
            try:
                normalized = root.resolve()
            except Exception:
                normalized = root
            if normalized in seen:
                continue
            seen.add(normalized)
            if root.exists():
                existing_roots.append(root)
        return existing_roots

    @classmethod
    def _resolve_cached_hf_dataset_path(cls, dataset_name: str) -> Optional[Path]:
        # HF datasets cache repo IDs under namespace___name in lowercase.
        if "/" not in dataset_name:
            return None

        cache_key = dataset_name.strip().lower().replace("/", "___")
        for cache_root in cls._hf_cache_roots():
            dataset_root = cache_root / cache_key
            if not dataset_root.exists():
                continue

            info_files = sorted(dataset_root.glob("**/dataset_info.json"))
            for info_file in info_files:
                candidate_dir = info_file.parent
                if any(candidate_dir.glob("*.arrow")):
                    return candidate_dir

            return dataset_root

        return None

    @staticmethod
    def _load_hf_dataset_dir(dataset_dir: Path, split: str):
        """Load either a load_from_disk dir or an HF cache Arrow-shard dir."""
        arrow_files = sorted(dataset_dir.glob("*.arrow"))
        if arrow_files:
            shard_datasets = [HFDataset.from_file(str(path)) for path in arrow_files]
            if len(shard_datasets) == 1:
                return shard_datasets[0]
            return concatenate_datasets(shard_datasets)

        dataset_obj = load_from_disk(str(dataset_dir))
        if isinstance(dataset_obj, DatasetDict):
            if split in dataset_obj:
                return dataset_obj[split]
            if "train" in dataset_obj:
                return dataset_obj["train"]
            first_split = str(next(iter(dataset_obj.keys())))
            return dataset_obj[first_split]
        return dataset_obj
    
    def __init__(
        self,
        data_source: Union[str, object],
        tokenizer: BaseTokenizer,
        seq_len: int,
        instruction_format: str = "auto",
        seed: int = 42,
        split: str = "train",
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        text_field: Optional[str] = None,
        pack_sequences: bool = False,
    ):
        """
        Args:
            data_source: Path to JSONL file, HF dataset name, or dataset object
            tokenizer: Tokenizer to use
            seq_len: Maximum sequence length
            instruction_format: Format (alpaca, chatml, sharegpt, auto)
            seed: Random seed for shuffling
            split: Dataset split to use (for HF datasets)
            hf_dataset_name: Explicitly specify HF dataset name
            text_field: Field containing text data (for simple datasets)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.instruction_format = instruction_format
        self.text_field = text_field
        self.pack_sequences = pack_sequences
        self.chat_template_tokenizer = self._resolve_chat_template_tokenizer(tokenizer)
        
        # Load data from various sources
        self.data: Union[list[dict[str, Any]], HFDataset] = []
        
        # Try to load as HuggingFace dataset first
        if hf_dataset_name or (isinstance(data_source, str) and not Path(data_source).exists()):
            from datasets import load_dataset

            dataset_name = str(hf_dataset_name or data_source)
            dataset_names_to_try = [dataset_name]
            if any(char.isupper() for char in dataset_name):
                lowered = dataset_name.lower()
                if lowered not in dataset_names_to_try:
                    dataset_names_to_try.append(lowered)

            hf_load_error: Optional[Exception] = None
            for candidate_name in dataset_names_to_try:
                try:
                    if hf_dataset_config:
                        logger.info(
                            f"Loading HuggingFace dataset: {candidate_name}/{hf_dataset_config}, split: {split}"
                        )
                        dataset = load_dataset(candidate_name, hf_dataset_config, split=split)
                    else:
                        logger.info(f"Loading HuggingFace dataset: {candidate_name}, split: {split}")
                        dataset = load_dataset(candidate_name, split=split)
                    self.data = dataset
                    logger.info(f"Loaded {len(self.data)} examples from HuggingFace dataset")
                    break
                except Exception as e:
                    hf_load_error = e

            if len(self.data) == 0:
                cache_path = self._resolve_cached_hf_dataset_path(dataset_name)
                if cache_path is not None:
                    try:
                        logger.info(f"Loading HuggingFace dataset from local cache path: {cache_path}")
                        cached_dataset = self._load_hf_dataset_dir(cache_path, split=split)
                        self.data = cached_dataset
                        logger.info(f"Loaded {len(self.data)} examples from local HF cache")
                    except Exception as e:
                        hf_load_error = e

            if len(self.data) == 0:
                if hf_load_error is not None:
                    logger.warning(f"Failed to load as HuggingFace dataset: {hf_load_error}")
                logger.info("Attempting to load as local file...")

            # Auto-detect format from first example
            if self.instruction_format == "auto" and len(self.data) > 0:
                self.instruction_format = detect_format(cast(dict, self.data[0]))
                logger.info(f"Auto-detected format: {self.instruction_format}")
        
        # Load from local JSONL file if not already loaded
        if len(self.data) == 0 and isinstance(data_source, str):
            data_file = Path(data_source)

            if data_file.exists() and data_file.is_dir():
                logger.info(f"Loading instruction data from local HuggingFace dataset dir: {data_source}")
                directory_dataset = self._load_hf_dataset_dir(data_file, split=split)
                self.data = directory_dataset
                logger.info(f"Loaded {len(self.data)} instruction examples from local dataset directory")
                if self.instruction_format == "auto" and len(self.data) > 0:
                    self.instruction_format = detect_format(cast(dict, self.data[0]))
                    logger.info(f"Auto-detected format: {self.instruction_format}")
            
            if not data_file.exists():
                raise FileNotFoundError(
                    f"Data file not found: {data_source}\n"
                    f"Tried as HuggingFace dataset and local file."
                )

            if len(self.data) == 0 and data_file.is_file():
                logger.info(f"Loading instruction data from {data_source}")

                if not isinstance(self.data, list):
                    self.data = []

                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(cast(dict[str, Any], json.loads(line)))

                logger.info(f"Loaded {len(self.data)} instruction examples from file")

                # Auto-detect format
                if self.instruction_format == "auto" and len(self.data) > 0:
                    self.instruction_format = detect_format(cast(dict, self.data[0]))
                    logger.info(f"Auto-detected format: {self.instruction_format}")
        
        if len(self.data) == 0:
            raise ValueError("No data loaded from any source")


        # Shuffle with seed for reproducibility only for in-memory Python lists.
        # HF datasets are already shuffled by DistributedSampler at training time.
        if isinstance(self.data, list):
            import random

            random.seed(seed)
            random.shuffle(self.data)
        else:
            logger.info("Using Arrow-backed dataset without Python list materialization/shuffle for faster startup")

    def _resolve_chat_template_tokenizer(self, tokenizer: BaseTokenizer) -> Optional[Any]:
        # 1) Directly support tokenizers that already expose apply_chat_template.
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer

        # 2) Support wrappers that expose an underlying tokenizer object.
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None and hasattr(inner, "apply_chat_template"):
            return inner

        # 3) Best effort: load HF tokenizer from tokenizer_path if available.
        tokenizer_path = getattr(tokenizer, "tokenizer_path", None)
        if tokenizer_path:
            try:
                transformers = importlib.import_module("transformers")
                AutoTokenizer = transformers.AutoTokenizer
                return AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
            except Exception as e:
                logger.debug(f"Could not load HF tokenizer for chat template from {tokenizer_path}: {e}")
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_alpaca(self, example: dict) -> tuple[str, str]:
        """
        Format Alpaca-style instruction.
        
        Format:
            Below is an instruction that describes a task...
            ### Instruction: {instruction}
            ### Input: {input}
            ### Response: {output}
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )
        
        return prompt, output
    
    def format_chatml(self, example: dict) -> tuple[str, str]:
        """
        Format ChatML-style conversation.
        
        Format:
            <|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {user}<|im_end|>
            <|im_start|>assistant
            {assistant}<|im_end|>
        """
        messages = example.get("messages") or []

        normalized_messages = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            normalized_messages.append({"role": role, "content": content})

        assistant_idx = None
        response = ""
        for i, msg in enumerate(normalized_messages):
            if msg["role"] == "assistant":
                assistant_idx = i
                response = msg["content"]
                break

        if assistant_idx is None:
            prompt_messages = normalized_messages
        else:
            prompt_messages = normalized_messages[:assistant_idx]

        has_template = bool(getattr(self.chat_template_tokenizer, "chat_template", None)) if self.chat_template_tokenizer is not None else False
        if self.chat_template_tokenizer is not None and hasattr(self.chat_template_tokenizer, "apply_chat_template") and has_template and prompt_messages:
            prompt = self.chat_template_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(prompt), response

        prompt_parts = []
        for msg in prompt_messages:
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        return prompt, response
    
    def format_sharegpt(self, example: dict) -> tuple[str, str]:
        """
        Format ShareGPT-style conversation.
        
        Format from conversations with 'from' and 'value' keys.
        """
        conversations = example.get("conversations", [])
        
        prompt_parts = []
        response = ""
        
        for conv in conversations:
            role = conv.get("from", "human")
            content = conv.get("value", "")
            
            if role in ["gpt", "assistant"]:
                response = content
                break
            elif role in ["human", "user"]:
                prompt_parts.append(f"User: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        return prompt, response
    
    def format_prompt_completion(self, example: dict) -> tuple[str, str]:
        """Format simple prompt-completion pairs."""
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        return prompt, completion
    
    def format_text(self, example: dict) -> tuple[str, str]:
        """
        Format plain text data by using text_field or 'text' key.
        For plain text, we don't mask anything (train on full text).
        """
        if self.text_field:
            text = example.get(self.text_field, "")
        else:
            text = example.get("text", "")
        
        # For plain text, we split arbitrarily or use whole text
        # Here we use empty prompt and full text as response
        return "", text
    
    def format_auto(self, example: dict) -> tuple[str, str]:
        """
        Auto-detect and format based on available fields.
        Tries common field names for question/answer pairs.
        """
        # Try common prompt field names
        prompt_fields = ["question", "query", "prompt", "input", "instruction"]
        response_fields = ["answer", "response", "output", "completion", "text"]
        
        prompt = ""
        response = ""
        
        for field in prompt_fields:
            if field in example and example[field]:
                prompt = str(example[field])
                break
        
        for field in response_fields:
            if field in example and example[field]:
                response = str(example[field])
                break
        
        if not response:
            # Fallback: use any text-like field
            for key, value in example.items():
                if isinstance(value, str) and value:
                    response = value
                    break
        
        return prompt, response
    
    def format_cot(self, example: dict) -> tuple[str, str]:
        """
        Format Chain-of-Thought (CoT) datasets.
        
        Combines problem/question with chain-of-thought reasoning.
        Common field names:
        - Problem: question, problem, query, prompt
        - CoT: chain_of_thought, thought, thinking, reasoning, explanation
        - Answer: answer, response, output, solution, final_answer
        """
        # Find problem
        problem_fields = ["question", "problem", "query", "prompt", "input", "instruction"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find chain-of-thought
        cot_fields = ["chain_of_thought", "thought", "thinking", "reasoning", "explanation", "steps"]
        cot = ""
        for field in cot_fields:
            if field in example and example[field]:
                cot_value = example[field]
                if isinstance(cot_value, list):
                    # Handle list of reasoning steps
                    cot = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(cot_value)])
                else:
                    cot = str(cot_value)
                break
        
        # Find answer
        answer_fields = ["answer", "response", "output", "solution", "final_answer", "conclusion"]
        answer = ""
        for field in answer_fields:
            if field in example and example[field]:
                answer = str(example[field])
                break
        
        # Combine: problem is prompt, cot + answer is response
        prompt = problem
        response = f"{cot}\n\nFinal Answer: {answer}" if cot and answer else answer
        
        return prompt, response
    
    def format_reasoning_steps(self, example: dict) -> tuple[str, str]:
        """
        Format multi-step reasoning datasets.
        
        Structure: problem -> intermediate reasoning steps -> final answer
        Common field names:
        - Problem: question, problem, query, context
        - Steps: steps, reasoning, intermediate_steps, process
        - Answer: answer, final_answer, solution, output
        """
        # Find problem/question
        problem_fields = ["question", "problem", "query", "context", "prompt", "input"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find reasoning steps
        step_fields = ["steps", "reasoning", "intermediate_steps", "process", "work"]
        steps = ""
        for field in step_fields:
            if field in example and example[field]:
                steps_value = example[field]
                if isinstance(steps_value, list):
                    steps = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps_value)])
                elif isinstance(steps_value, str):
                    steps = steps_value
                break
        
        # Find final answer
        answer_fields = ["answer", "final_answer", "solution", "output", "result"]
        answer = ""
        for field in answer_fields:
            if field in example and example[field]:
                answer = str(example[field])
                break
        
        # Combine: problem is prompt, steps + answer is response
        prompt = problem
        if steps and answer:
            response = f"{steps}\n\nFinal Answer: {answer}"
        elif steps:
            response = steps
        else:
            response = answer
        
        return prompt, response
    
    def format_problem_solution(self, example: dict) -> tuple[str, str]:
        """
        Format problem-solution datasets.
        
        Common field names:
        - Problem: problem, question, query, task
        - Solution: solution, answer, code, output, response
        - Explanation: explanation, reasoning, description (optional)
        """
        # Find problem
        problem_fields = ["problem", "question", "query", "task", "prompt", "input"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find solution
        solution_fields = ["solution", "answer", "code", "output", "response", "result"]
        solution = ""
        for field in solution_fields:
            if field in example and example[field]:
                solution = str(example[field])
                break
        
        # Find optional explanation
        explanation_fields = ["explanation", "reasoning", "description", "notes"]
        explanation = ""
        for field in explanation_fields:
            if field in example and example[field]:
                explanation = str(example[field])
                break
        
        # Combine problem as prompt, solution (+ explanation) as response
        prompt = problem
        if explanation:
            response = f"{solution}\n\nExplanation: {explanation}"
        else:
            response = solution
        
        return prompt, response
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single example with input tokens, labels, and loss mask.
        
        Returns:
            Dictionary with:
                - input: Input token IDs (prompt + response)
                - labels: Target token IDs (shifted by 1)
                - loss_mask: Binary mask (1 for response tokens, 0 for prompt)
        """
        raw_example = self.data[idx]
        if isinstance(raw_example, dict):
            example = cast(dict[str, Any], raw_example)
        else:
            example = cast(dict[str, Any], dict(raw_example))

        # Format based on instruction format
        if self.instruction_format == "alpaca":
            prompt, response = self.format_alpaca(example)
        elif self.instruction_format == "chatml":
            prompt, response = self.format_chatml(example)
        elif self.instruction_format == "sharegpt":
            prompt, response = self.format_sharegpt(example)
        elif self.instruction_format == "prompt_completion":
            prompt, response = self.format_prompt_completion(example)
        elif self.instruction_format == "text":
            prompt, response = self.format_text(example)
        elif self.instruction_format == "cot":
            prompt, response = self.format_cot(example)
        elif self.instruction_format == "reasoning_steps":
            prompt, response = self.format_reasoning_steps(example)
        elif self.instruction_format == "problem_solution":
            prompt, response = self.format_problem_solution(example)
        elif self.instruction_format == "auto":
            prompt, response = self.format_auto(example)
        else:
            raise ValueError(f"Unknown instruction format: {self.instruction_format}")
        
        # Tokenize prompt and response separately
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        response_tokens = self.tokenizer.encode(response, bos=False, eos=True)
        
        # EOS token is now added by the tokenizer's encode method above
        
        # Combine and truncate to seq_len + 1 (to account for next-token prediction shift)
        # This matches standard pre-training: we need seq_len+1 tokens to create seq_len input and seq_len labels
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        if len(full_tokens) > self.seq_len + 1:
            # Truncate from the end (response)
            full_tokens = full_tokens[:self.seq_len + 1]
            prompt_len = min(prompt_len, self.seq_len + 1)
        
        # Ensure we can always create at least one next-token pair.
        if len(full_tokens) < 2:
            full_tokens = full_tokens + [self.tokenizer.eos_id]
        
        # Create loss mask: 1 for response tokens, 0 for prompt and padding
        valid_len = len(full_tokens)
        loss_mask = [0] * valid_len
        for i in range(prompt_len, valid_len):
            loss_mask[i] = 1

        if not self.pack_sequences:
            # Pad if necessary to seq_len + 1 in non-packing mode.
            padding_len = (self.seq_len + 1) - len(full_tokens)
            if padding_len > 0:
                _pad_id = self.tokenizer.pad_id
                if _pad_id is None or _pad_id < 0:
                    _pad_id = self.tokenizer.eos_id
                full_tokens = full_tokens + [_pad_id] * padding_len
                loss_mask = loss_mask + [0] * padding_len
        
        # Convert to tensors - use next-token prediction shift
        # input_ids: tokens 0..seq_len-1 (drop last token)
        # labels: tokens 1..seq_len (drop first token)
        # This gives us seq_len input tokens and seq_len label tokens
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)  # Remove last token for input
        labels = torch.tensor(full_tokens[1:], dtype=torch.long)  # Shift by 1 for labels
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Align with labels
        
        return {
            "input": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def build_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a dataloader for supervised fine-tuning with instruction data.
    
    Supports:
    - HuggingFace datasets: Set data.hf_dataset_name or use data_prefix as HF dataset
    - Local JSONL files: Set data_prefix to file path
    - Auto format detection or explicit format setting
    
    Config options:
    - data.data_prefix: Path to JSONL file or HF dataset name
    - data.hf_dataset_name: Explicitly specify HF dataset (optional)
    - data.instruction_format: Format (alpaca, chatml, sharegpt, auto)
    - data.dataset_split: Split to use for HF datasets (default: train)
    - data.text_field: Field name for plain text datasets (optional)
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer instance
        job_config: Job configuration
        infinite: Whether to loop infinitely over the dataset
    
    Returns:
        ParallelAwareDataloader for SFT
    """
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    data_source = job_config.data.data_prefix
    instruction_format = getattr(job_config.data, 'instruction_format', 'auto')
    seed = job_config.data.seed
    
    # HuggingFace dataset specific options
    hf_dataset_name = getattr(job_config.data, 'hf_dataset_name', None)
    hf_dataset_config = getattr(job_config.data, 'hf_dataset_config', None)
    dataset_split = getattr(job_config.data, 'dataset_split', 'train')
    text_field = getattr(job_config.data, 'text_field', None)
    prepacked_dataset = bool(getattr(job_config.data, 'prepacked_dataset', False))
    packing_strategy = getattr(job_config.data, 'packing_strategy', 'none').lower()
    if packing_strategy not in {'none', 'obfd', 'sequential'}:
        raise ValueError(
            f"Unknown SFT packing_strategy: {packing_strategy!r}. "
            "Expected 'none', 'obfd', or 'sequential'."
        )
    if prepacked_dataset and packing_strategy in {'obfd', 'sequential'}:
        logger.warning(
            "prepacked_dataset=true with packing_strategy=%s; packing will be disabled.",
            packing_strategy,
        )
        packing_strategy = 'none'
    use_obfd       = packing_strategy == 'obfd'
    use_sequential = packing_strategy == 'sequential'

    # Attention config — needed for sequential packing collate selection.
    attn_type       = getattr(job_config.model, 'attn_type', 'sdpa')
    attn_mask_type  = getattr(job_config.model, 'attn_mask_type', 'causal')
    use_doc_masking = use_sequential and attn_mask_type == 'block_causal'
    # varlen/ring_varlen need cu_seqlens from the collate; SDPA uses a 4D mask instead.
    use_varlen_collate = attn_type in ('varlen', 'ring_varlen')

    logger.info(
        f"Building SFT dataloader: "
        f"source={hf_dataset_name or data_source}, "
        f"format={instruction_format}, "
        f"packing={packing_strategy}, "
        f"attn={attn_type}, "
        f"attn_mask={attn_mask_type}, "
        f"batch_size={batch_size}, "
        f"seq_len={seq_len}"
    )

    # Create dataset
    if prepacked_dataset:
        if not isinstance(data_source, str) or not Path(data_source).exists():
            raise RuntimeError(
                "prepacked_dataset=true requires data.data_prefix to be a local load_from_disk path"
            )
        dataset = PrepackedInstructionDataset(data_source)
        collate_fn = _collate_prepacked
    else:
        raw_dataset = InstructionDataset(
            data_source=data_source,
            tokenizer=tokenizer,
            seq_len=seq_len,
            instruction_format=instruction_format,
            seed=seed,
            split=dataset_split,
            hf_dataset_name=hf_dataset_name,
            hf_dataset_config=hf_dataset_config,
            text_field=text_field,
            pack_sequences=use_obfd,
        )
        pad_id = tokenizer.pad_id if tokenizer.pad_id is not None and tokenizer.pad_id >= 0 else tokenizer.eos_id
        if use_sequential:
            logger.info(
                "packing_strategy=sequential (OLMo3-style greedy): "
                "scanning all documents at startup to build packing index"
            )
            index_cache_path = getattr(job_config.data, 'sequential_index_cache_path', None) or None
            dataset = SequentialPackedSFTDataset(
                raw_dataset, seq_len=seq_len, pad_id=pad_id, index_cache_path=index_cache_path
            )
            if use_doc_masking:
                min_doc_len = getattr(job_config.data, 'min_doc_len', 1)
                max_cu_seqlens_size = (
                    batch_size * (seq_len // max(1, min_doc_len) + 2) + 1
                    if use_varlen_collate else None
                )
                collate_fn = lambda batch, _seq=seq_len, _pad=pad_id, _vc=use_varlen_collate, _mc=max_cu_seqlens_size: (
                    _collate_sft_doc_mask(batch, seq_len=_seq, pad_id=_pad,
                                          use_flash_attention=_vc, max_cu_seqlens_size=_mc)
                )
            else:
                # sequential packing without block-causal: use standard causal attention
                collate_fn = _collate_prepacked
        elif use_obfd:
            logger.info(
                "packing_strategy=obfd (online global): building one-time packing plan before training starts; "
                "startup may be slow for large datasets"
            )
            dataset = GlobalOBFDPackedDataset(raw_dataset, seq_len=seq_len, pad_id=pad_id)
            collate_fn = _collate_prepacked
        else:
            dataset = raw_dataset
            collate_fn = lambda batch: _collate_pad(batch, pad_id=pad_id)
    
    # Create dataloader with proper sharding for distributed training
    from torch.utils.data import DistributedSampler
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )
    
    # Create ParallelAwareDataloader for TorchTitan compatibility
    parallel_dataloader = ParallelAwareDataloader(
        dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Can be increased for faster data loading
        pin_memory=True,
    )
    
    logger.info(f"SFT dataloader created with {len(dataset)} examples")
    
    return parallel_dataloader
