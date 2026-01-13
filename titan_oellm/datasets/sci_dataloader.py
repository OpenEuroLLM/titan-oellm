import logging
import math
from functools import partial
from pathlib import Path

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig

from titan_oellm.constants import IGNORE_INDEX
from titan_oellm.datasets.dataloader.mmap_dataset import MMapDataset
# from titan_oellm.datasets.dataloader.parallel_mmap_dataset_chunked import ChunkedMMapDataset
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
    if getattr(job_config.validation, 'data_source', 'offline') != 'split':
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

    logger.info(f"Split config: {split_samples} requested → {docs_per_chunk} docs/chunk × {num_chunks} chunks = {actual_samples} actual")
    return docs_per_chunk




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
    chunks_dir = job_config.data.chunks_dir
    dataloader = job_config.data.dataloader
    seed = job_config.data.seed

    ignore_index = IGNORE_INDEX

    # Check for validation split mode - training excludes validation samples
    exclude_last_n = None           # For MMapDataset
    exclude_first_n_per_chunk = None   # For ChunkedMMapDataset

    if (hasattr(job_config, 'validation') and
        getattr(job_config.validation, 'data_source', 'offline') == 'split'):

        if dataloader == "MMapDataset":
            exclude_last_n = job_config.validation.split_samples
            if exclude_last_n > 0:
                logger.info(f"Training dataloader: excluding last {exclude_last_n} samples for validation split")
        elif dataloader == "ChunkedMMapDataset":
            exclude_first_n_per_chunk = _compute_docs_per_chunk(job_config)
            if exclude_first_n_per_chunk > 0:
                logger.info(f"Training dataloader: excluding first {exclude_first_n_per_chunk} docs per chunk for validation split")

    if dataloader == "MMapDataset":
        dataset = MMapDataset(
            path_prefix=data_prefix,
            shuffle=True,
            infinite=infinite,
            limit_samples=False,
            seed=seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            validate=True,  # Enable automatic validation
            exclude_last_n=exclude_last_n,
        )
    elif dataloader == "ChunkedMMapDataset":
        dataset = ChunkedMMapDataset(
            chunks_dir=chunks_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            infinite=True,
            seed=seed,
            exclude_first_n_per_chunk=exclude_first_n_per_chunk,
        )
    else:
        raise ValueError(f"Unknown dataloader: {dataloader}")

    # Pass eos_id for document masking when block_causal attention is enabled
    # eos_id = None
    # if hasattr(job_config.model, 'attn_mask_type') and job_config.model.attn_mask_type == "block_causal":
    eos_id = tokenizer.eos_id

    sequencer = StreamingSequencer(
        dataset=dataset,
        sequence_length=seq_len,
        min_sequence_length=min_doc_len,
        drop_last=True,
        eos_id=eos_id)

    use_flash_attention = getattr(job_config.model, 'use_flash_attn', False)

    # Fixed cu_seqlens size for torch.compile fullgraph mode
    # Max docs = batch_size * (max docs per sample) + 1 for the leading zero
    max_cu_seqlens_size = batch_size * (seq_len // min_doc_len + 1) + 1 if use_flash_attention else None

    collate_fn = partial(
        collate_function,
        ignore_index=ignore_index,
        use_flash_attention=use_flash_attention,
        eos_id=eos_id,
        seq_len=seq_len,
        max_cu_seqlens_size=max_cu_seqlens_size,
    )

    dataloader = ParallelAwareDataloader(
        dataset=sequencer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    # Add ignore_index as attribute for use in loss calculation
    dataloader.ignore_index = ignore_index
    return dataloader


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
    dataloader = job_config.validation.dataloader
    seed = job_config.data.seed

    ignore_index = IGNORE_INDEX

    # Calculate limit_samples from max_eval_samples config
    # -1 means use full validation set, >0 means limit to N samples total across all workers
    max_eval_samples = job_config.validation.max_eval_samples
    limit_samples = False if max_eval_samples == -1 else max_eval_samples

    # Determine data source and split params
    data_source = getattr(job_config.validation, 'data_source', 'offline')

    use_only_last_n = None              # For MMapDataset
    use_only_first_n_per_chunk = None   # For ChunkedMMapDataset

    if data_source == 'split':
        # Use same data source as training, with split
        if dataloader == "MMapDataset":
            data_prefix = job_config.data.data_prefix  # Use training data path
            use_only_last_n = job_config.validation.split_samples
            logger.info(f"Validation dataloader (split mode): using last {use_only_last_n} samples from training data")
        elif dataloader == "ChunkedMMapDataset":
            chunks_dir = job_config.data.chunks_dir  # Use training chunks dir
            use_only_first_n_per_chunk = _compute_docs_per_chunk(job_config)
            logger.info(f"Validation dataloader (split mode): using first {use_only_first_n_per_chunk} docs per chunk")
        else:
            raise ValueError(f"Unknown validation dataloader: {dataloader}")
    else:
        # Offline mode: use separate validation data
        data_prefix = job_config.validation.data_prefix
        chunks_dir = job_config.validation.data_prefix  # For ChunkedMMapDataset, use data_prefix as chunks_dir

    if dataloader == "MMapDataset":
        dataset = MMapDataset(
            path_prefix=data_prefix,
            shuffle=False,  # Don't shuffle validation data
            infinite=False,  # Don't loop validation data infinitely
            limit_samples=limit_samples,  # Limit total samples before worker partitioning
            seed=seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            validate=True,  # Enable automatic validation
            use_only_last_n=use_only_last_n,
        )
    elif dataloader == "ChunkedMMapDataset":
        dataset = ChunkedMMapDataset(
            chunks_dir=chunks_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            infinite=False,  # Don't loop validation data infinitely
            seed=seed,
            use_only_first_n_per_chunk=use_only_first_n_per_chunk,
        )
    else:
        raise ValueError(f"Unknown validation dataloader: {dataloader}")

    eos_id = tokenizer.eos_id
    pad_id = getattr(tokenizer, 'pad_id', 0)  # Fall back to 0 if no pad_id

    # Check evaluation mode
    eval_mode = getattr(job_config.validation, 'eval_mode', 'concatenated')

    if eval_mode == 'document':
        # Document mode: skip sequencer, evaluate each document independently
        # Documents are padded/truncated directly without concatenation
        logger.info("Validation using document mode: each document evaluated independently")

        collate_fn = partial(
            collate_function_document_eval,
            seq_len=seq_len,
            ignore_index=ignore_index,
            pad_id=pad_id,
        )

        dataloader_dataset = dataset  # Use dataset directly, no sequencer

    else:
        # Concatenated mode (default): use sequencer to pack documents
        sequencer = StreamingSequencer(
            dataset=dataset,
            sequence_length=seq_len,
            min_sequence_length=min_doc_len,
            drop_last=False,  # Don't drop last for validation to avoid uneven batches across ranks
            eos_id=eos_id)

        use_flash_attention = getattr(job_config.model, 'use_flash_attn', False)

        # Fixed cu_seqlens size for torch.compile fullgraph mode
        # Use training batch size to ensure same size as training dataloader
        train_batch_size = job_config.training.local_batch_size
        max_cu_seqlens_size = train_batch_size * (seq_len // min_doc_len + 1) + 1 if use_flash_attention else None

        collate_fn = partial(
            collate_function,
            ignore_index=ignore_index,
            use_flash_attention=False,
            eos_id=eos_id,
            seq_len=seq_len,
            max_cu_seqlens_size=max_cu_seqlens_size,
        )

        dataloader_dataset = sequencer

    dataloader = ParallelAwareDataloader(
        dataset=dataloader_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    # Add ignore_index as attribute for use in validation loss calculation
    dataloader.ignore_index = ignore_index
    return dataloader

