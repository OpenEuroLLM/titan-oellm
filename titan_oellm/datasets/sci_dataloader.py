import logging
import math
from functools import partial
from pathlib import Path

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig

from titan_oellm.constants import IGNORE_INDEX
from titan_oellm.datasets.dataloader.mmap_dataset import MMapDataset
from titan_oellm.datasets.dataloader.deterministic_packed_dataset import DeterministicPackedDataset
from titan_oellm.datasets.dataloader.mmap_dataset_chunked import ChunkedMMapDataset
from titan_oellm.datasets.sequencer.simple_concat import StreamingSequencer
from titan_oellm.datasets.utils.collator import collate_function, collate_function_document_eval

logger = logging.getLogger(__name__)


def _compute_docs_per_chunk(job_config: JobConfig) -> int:
    """Compute docs per chunk for validation split (rounds up to ensure >= requested samples).

    For ChunkedMMapDataset split mode: takes first K docs from each chunk for validation.
    This ensures worker-count independence - same validation set regardless of worker count.

    Returns:
        Number of docs per chunk to use for validation split, or 0 if not in split mode.
    """
    if getattr(job_config.validation, "data_source", "offline") != "split":
        return 0

    split_samples = job_config.validation.split_samples
    if split_samples <= 0:
        return 0

    # Count chunks
    chunks_dir = Path(job_config.data.chunks_dir)
    if not chunks_dir.exists():
        raise ValueError(f"Chunks directory not found: {chunks_dir}")

    num_chunks = len(list(chunks_dir.glob("chunk_*.idx")))
    if num_chunks == 0:
        raise ValueError(f"No chunks found in {chunks_dir}")

    # Round up to ensure we get at least split_samples
    docs_per_chunk = math.ceil(split_samples / num_chunks)
    actual_samples = docs_per_chunk * num_chunks

    logger.info(
        f"Split config: {split_samples} requested → {docs_per_chunk} docs/chunk × {num_chunks} chunks = {actual_samples} actual"
    )
    return docs_per_chunk




def _resolve_attention_config(job_config: JobConfig, batch_size: int, seq_len: int, min_doc_len: int) -> dict:
    """Resolve flash/flex attention settings from model config.

    Returns dict with use_flash_attention, use_document_mask, max_cu_seqlens_size.
    """
    use_flash_attention = getattr(job_config.model, "use_flash_attn", False)
    use_flex_attn = getattr(job_config.model, "use_flex_attn", False)
    attn_mask_type = getattr(job_config.model, "attn_mask_type", "causal")

    # block_causal must go through FlexAttention — SDPA+mask falls back to math backend
    if attn_mask_type == "block_causal":
        use_flex_attn = True

    use_document_mask = attn_mask_type == "block_causal" and not use_flex_attn and not use_flash_attention

    # Fixed cu_seqlens size for torch.compile fullgraph mode
    # Max docs = batch_size * (max docs per sample) + 1 for the leading zero
    max_cu_seqlens_size = batch_size * (seq_len // min_doc_len + 1) + 1 if use_flash_attention else None

    return {
        "use_flash_attention": use_flash_attention,
        "use_document_mask": use_document_mask,
        "max_cu_seqlens_size": max_cu_seqlens_size,
    }


def _make_collate_fn(
    dataloader_type: str,
    attn_config: dict,
    eos_id,
    seq_len: int,
    ignore_index: int,
    mask_eot_loss: bool = False,
) -> partial:
    """Build the collate function for training/concatenated-validation pipelines."""
    return partial(
        collate_function,
        ignore_index=ignore_index,
        use_flash_attention=attn_config["use_flash_attention"],
        use_document_mask=attn_config["use_document_mask"],
        eos_id=eos_id,
        seq_len=seq_len,
        max_cu_seqlens_size=attn_config["max_cu_seqlens_size"],
        mask_eot_loss=mask_eot_loss,
    )


def build_sci_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for Sci datasets."""
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    min_doc_len = job_config.data.min_doc_len
    data_prefix = job_config.data.data_prefix
    dataloader_type = job_config.data.dataloader
    seed = job_config.data.seed

    # For chunk-based dataloaders, data_prefix doubles as the chunk directory source
    # when chunks_dir is not explicitly set.
    chunks_dir = job_config.data.chunks_dir or data_prefix

    ignore_index = IGNORE_INDEX

    # Resolve effective eos_id: -1 → use tokenizer value; None → no EOS; >=0 → explicit
    cfg_eos = getattr(job_config.data, "eos_id", -1)
    eos_id: int | None = tokenizer.eos_id if cfg_eos == -1 else cfg_eos

    # Check for validation split mode — training excludes validation samples
    exclude_last_n = None  # For MMapDataset
    exclude_first_n_per_chunk = None  # For ChunkedMMapDataset

    if hasattr(job_config, "validation") and getattr(job_config.validation, "data_source", "offline") == "split":
        if dataloader_type == "MMapDataset":
            exclude_last_n = job_config.validation.split_samples
            if exclude_last_n > 0:
                logger.info(f"Training dataloader: excluding last {exclude_last_n} samples for validation split")
        elif dataloader_type == "ChunkedMMapDataset":
            exclude_first_n_per_chunk = _compute_docs_per_chunk(job_config)
            if exclude_first_n_per_chunk > 0:
                logger.info(
                    f"Training dataloader: excluding first {exclude_first_n_per_chunk} docs per chunk for validation split"
                )

    # ── 1. Create dataset ────────────────────────────────────────────────

    if dataloader_type == "MMapDataset":
        prefixes = data_prefix if isinstance(data_prefix, list) else [data_prefix]
        if len(prefixes) == 1:
            dataset = MMapDataset(
                path_prefix=prefixes[0],
                shuffle=True,
                infinite=infinite,
                limit_samples=False,
                seed=seed,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                validate=True,
                exclude_last_n=exclude_last_n,
            )
        else:
            import torch
            datasets = [
                MMapDataset(
                    path_prefix=p,
                    shuffle=True,
                    infinite=infinite,
                    limit_samples=False,
                    seed=seed,
                    dp_world_size=dp_world_size,
                    dp_rank=dp_rank,
                    validate=True,
                    exclude_last_n=exclude_last_n,
                )
                for p in prefixes
            ]
            dataset = torch.utils.data.ConcatDataset(datasets)
    elif dataloader_type == "ChunkedMMapDataset":
        dataset = ChunkedMMapDataset(
            chunks_dir=chunks_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            infinite=True,
            seed=seed,
            exclude_first_n_per_chunk=exclude_first_n_per_chunk,
        )
    elif dataloader_type == "DeterministicPackedDataset":
        raw_gbs = job_config.training.global_batch_size
        resolved_gbs = raw_gbs if raw_gbs > 0 else job_config.training.local_batch_size * dp_world_size
        dataset = DeterministicPackedDataset(
            chunks_dir=chunks_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            global_batch_size=resolved_gbs,
            seq_len=seq_len,
            min_sequence_length=min_doc_len,
            eos_id=eos_id,
            infinite=True,
            seed=seed,
            exclude_first_n_per_chunk=exclude_first_n_per_chunk,
        )
    else:
        raise ValueError(f"Unknown dataloader: {dataloader_type}")

    # ── 2. Route to pipeline ─────────────────────────────────────────────
    #
    # Pre-packed datasets (DeterministicPackedDataset, BestFitPackedDataset)
    # yield packed sequences — no sequencer needed.
    # All other datasets yield documents that need StreamingSequencer.

    if dataloader_type in ("DeterministicPackedDataset", "BestFitPackedDataset"):
        wrapped_dataset = dataset
    else:
        sequencer = StreamingSequencer(
            dataset=dataset, sequence_length=seq_len, min_sequence_length=min_doc_len, drop_last=True, eos_id=eos_id
        )
        wrapped_dataset = sequencer

    # ── 3. Collate and wrap ──────────────────────────────────────────────

    attn_config = _resolve_attention_config(job_config, batch_size, seq_len, min_doc_len)
    mask_eot_loss = getattr(job_config.data, "mask_eot_loss", False)
    collate_fn = _make_collate_fn(dataloader_type, attn_config, eos_id, seq_len, ignore_index, mask_eot_loss)

    dl = ParallelAwareDataloader(
        dataset=wrapped_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    dl.ignore_index = ignore_index
    return dl


# ── Validation dataloader ───────────────────────────────────────────────────


def build_sci_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
) -> ParallelAwareDataloader:
    """Build a validation data loader for Sci datasets."""
    # Use training batch size if validation batch size is not specified
    batch_size = (
        job_config.validation.local_batch_size
        if job_config.validation.local_batch_size > 0
        else job_config.training.local_batch_size
    )
    seq_len = job_config.training.seq_len
    min_doc_len = job_config.data.min_doc_len
    dataloader_type = job_config.validation.dataloader
    seed = job_config.data.seed

    ignore_index = IGNORE_INDEX

    # Calculate limit_samples from max_eval_samples config
    # -1 means use full validation set, >0 means limit to N samples total across all workers
    max_eval_samples = job_config.validation.max_eval_samples
    limit_samples = False if max_eval_samples == -1 else max_eval_samples

    # Determine data source and split params
    data_source = getattr(job_config.validation, "data_source", "offline")

    use_only_last_n = None  # For MMapDataset
    use_only_first_n_per_chunk = None  # For ChunkedMMapDataset

    if data_source == "split":
        # Use same data source as training, with split
        if dataloader_type == "MMapDataset":
            data_prefix = job_config.data.data_prefix  # Use training data path
            use_only_last_n = job_config.validation.split_samples
            logger.info(f"Validation dataloader (split mode): using last {use_only_last_n} samples from training data")
        elif dataloader_type == "ChunkedMMapDataset":
            chunks_dir = job_config.data.chunks_dir  # Use training chunks dir
            use_only_first_n_per_chunk = _compute_docs_per_chunk(job_config)
            logger.info(f"Validation dataloader (split mode): using first {use_only_first_n_per_chunk} docs per chunk")
        else:
            raise ValueError(f"Unknown validation dataloader: {dataloader_type}")
    else:
        # Offline mode: use separate validation data
        data_prefix = job_config.validation.data_prefix
        chunks_dir = job_config.validation.data_prefix  # For ChunkedMMapDataset, use data_prefix as chunks_dir

    # ── 1. Create dataset ────────────────────────────────────────────────

    if dataloader_type == "MMapDataset":
        dataset = MMapDataset(
            path_prefix=data_prefix,
            shuffle=False,
            infinite=False,
            limit_samples=limit_samples,
            seed=seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            validate=True,
            use_only_last_n=use_only_last_n,
        )
    elif dataloader_type == "ChunkedMMapDataset":
        dataset = ChunkedMMapDataset(
            chunks_dir=chunks_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            infinite=False,
            seed=seed,
            use_only_first_n_per_chunk=use_only_first_n_per_chunk,
        )
    else:
        raise ValueError(f"Unknown validation dataloader: {dataloader_type}")

    # ── 2. Route to pipeline ─────────────────────────────────────────────

    cfg_eos = getattr(job_config.data, "eos_id", -1)
    eos_id: int | None = tokenizer.eos_id if cfg_eos == -1 else cfg_eos
    pad_id = getattr(tokenizer, "pad_id", 0)
    if pad_id is None or pad_id < 0:
        pad_id = eos_id

    eval_mode = getattr(job_config.validation, "eval_mode", "concatenated")

    if eval_mode == "document":
        # Document mode: evaluate each document independently (no sequencer).
        # Each sample = one document, truncated/padded to seq_len.
        # This ensures document-aware validation regardless of training masking:
        # no cross-document attention is possible with single-document samples.
        logger.info("Validation using document mode: each document evaluated independently")

        collate_fn = partial(
            collate_function_document_eval,
            seq_len=seq_len,
            ignore_index=ignore_index,
            pad_id=pad_id,
        )
        wrapped_dataset = dataset

    else:
        # Concatenated mode (default): pack documents via sequencer
        sequencer = StreamingSequencer(
            dataset=dataset,
            sequence_length=seq_len,
            min_sequence_length=min_doc_len,
            drop_last=False,
            eos_id=eos_id,
        )

        attn_config = _resolve_attention_config(
            job_config, job_config.training.local_batch_size, seq_len, min_doc_len
        )
        # Validation disables flash attention for simplicity
        attn_config["use_flash_attention"] = False

        mask_eot_loss = getattr(job_config.data, "mask_eot_loss", False)
        collate_fn = _make_collate_fn(dataloader_type, attn_config, eos_id, seq_len, ignore_index, mask_eot_loss)
        wrapped_dataset = sequencer

    # ── 3. Wrap and return ───────────────────────────────────────────────

    dl = ParallelAwareDataloader(
        dataset=wrapped_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    dl.ignore_index = ignore_index
    return dl
