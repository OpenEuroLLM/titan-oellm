"""Prebuild BestFitPackedDataset packing plans on the login node.

Reuses torchtitan's ConfigManager so the script accepts exactly the same CLI
flags as the training entry-point (`python -m torchtitan.train ...`). The
BFP cache_key is deterministic in those flags, so a plan built here will be
loaded cache-hit by the multi-node sbatch job that follows.

Submit_job.sh runs this automatically before sbatch / local exec when
--data.dataloader=BestFitPackedDataset is in the args. Skip via env var
SKIP_BFP_PREBUILD=1.

Usage:

    python -m titan_oellm.datasets.utils.prebuild_bfp \
        --job.config_file=... --data.dataloader=BestFitPackedDataset \
        --data.chunks_dir=... --training.seq_len=4096 ...

Exit codes:
    0  — cache existed (hit) or build succeeded
    1  — build failed
    2  — args don't request BestFitPackedDataset (no-op)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import math

from torchtitan.config import ConfigManager

from titan_oellm.datasets.dataloader.bestfit_packed_dataset import BestFitPackedDataset
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import SciHFTokenizer

logger = logging.getLogger("prebuild_bfp")


def _compute_docs_per_chunk(config) -> int:
    """Mirror of titan_oellm.datasets.sci_dataloader._compute_docs_per_chunk.

    Inlined here to avoid importing the dataloader module (which pulls in torch
    and other deps the login-node prebuild script doesn't need).
    """
    if getattr(config.validation, "data_source", "offline") != "split":
        return 0
    split_samples = config.validation.split_samples
    if split_samples <= 0:
        return 0
    chunks_dir = Path(config.data.chunks_dir)
    if not chunks_dir.exists():
        return 0
    num_chunks = len(list(chunks_dir.glob("chunk_*.idx")))
    if num_chunks == 0:
        return 0
    return math.ceil(split_samples / num_chunks)


def _resolve_split_params(config) -> tuple[int | None, int | None]:
    """Match what build_dataloader does for training vs validation.

    Training: `exclude_first_n_per_chunk` is set when
    `validation.data_source == 'split'` and `split_samples > 0`.
    Validation (separate prebuild): `use_only_first_n_per_chunk` mirror.
    """
    try:
        n = _compute_docs_per_chunk(config)
    except Exception as exc:
        logger.warning(
            "Failed to compute split docs/chunk (%s); assuming no split", exc
        )
        n = 0
    return (n or None, None)


def _resolve_eos_id(config) -> int | None:
    """Build the same SciHFTokenizer the trainer would, just to read eos_id.

    Avoids touching torch — SciHFTokenizer wraps an HF tokenizer load only.
    """
    tokenizer_path = (
        getattr(config.model, "hf_assets_path", None)
        or config.model.tokenizer_path
    )
    if not tokenizer_path:
        raise RuntimeError(
            "No tokenizer path resolved (model.hf_assets_path / "
            "model.tokenizer_path). submit_job.sh normally injects this from "
            "cluster_paths.toml — check $TOKENIZER and $CLUSTER are set."
        )
    return SciHFTokenizer(tokenizer_path).eos_id


def _check_cache(ds: BestFitPackedDataset, epoch: int = 0) -> Path:
    cache_dir = ds._resolve_cache_dir()
    cache_key = ds._compute_bestfit_cache_key(epoch)
    return cache_dir / f"bestfit_{cache_key}.manifest.json"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        force=True,
    )

    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    if config.data.dataloader != "BestFitPackedDataset":
        logger.info(
            "dataloader=%s, not BestFitPackedDataset; nothing to prebuild",
            config.data.dataloader,
        )
        return 2

    chunks_dir = config.data.chunks_dir
    if not chunks_dir or not Path(chunks_dir).exists():
        logger.error("chunks_dir not set or missing: %r", chunks_dir)
        return 1

    eos_id = _resolve_eos_id(config)
    exclude_first_n, use_only_first_n = _resolve_split_params(config)

    raw_gbs = config.training.global_batch_size
    resolved_gbs = (
        raw_gbs if raw_gbs and raw_gbs > 0 else max(1, config.training.local_batch_size)
    )

    cache_dir_override = config.data.bfp_cache_dir or None

    logger.info("=" * 64)
    logger.info("BestFitPackedDataset prebuild")
    logger.info("  chunks_dir   = %s", chunks_dir)
    logger.info("  cache_dir    = %s", cache_dir_override or f"{chunks_dir}/.packing_cache")
    logger.info("  seq_len      = %s", config.training.seq_len)
    logger.info("  seed         = %s", config.data.seed)
    logger.info("  min_doc_len  = %s", config.data.min_doc_len)
    logger.info("  eos_id       = %s", eos_id)
    logger.info("  buffer_size  = %s", config.data.best_fit_buffer_size)
    logger.info("  exclude_first_n_per_chunk = %s", exclude_first_n)
    logger.info("=" * 64)

    # Ensure the cache dir exists *before* peeking at the plan manifest.
    cache_root = (
        Path(cache_dir_override) if cache_dir_override
        else Path(chunks_dir) / ".packing_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        ds = BestFitPackedDataset(
            chunks_dir=chunks_dir,
            dp_world_size=1,
            dp_rank=0,
            global_batch_size=resolved_gbs,
            seq_len=config.training.seq_len,
            min_sequence_length=config.data.min_doc_len,
            eos_id=eos_id,
            infinite=True,
            seed=config.data.seed,
            exclude_first_n_per_chunk=exclude_first_n,
            use_only_first_n_per_chunk=use_only_first_n,
            best_fit_buffer_size=config.data.best_fit_buffer_size,
            cache_dir=cache_dir_override,
        )
    except Exception:
        logger.exception("BestFitPackedDataset construction failed")
        return 1

    elapsed = time.monotonic() - t0
    cache_file = _check_cache(ds, epoch=0)
    logger.info(
        "OK: %d sequences, %d total tokens (cache: %s) in %.1fs",
        ds.total_sequences,
        ds.total_effective_tokens,
        cache_file,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
