#!/usr/bin/env python3
"""Download and preprocess standard benchmark datasets for LLM evaluation.

Downloads WikiText-2, WikiText-103, and LAMBADA from HuggingFace,
tokenizes them using SciHFTokenizer from cluster_paths.toml, and stores
in bin/idx format compatible with MMapDataset.

Output structure:
    {output-dir}/wikitext2/wikitext2.bin
    {output-dir}/wikitext2/wikitext2.idx
    {output-dir}/wikitext103/wikitext103.bin
    {output-dir}/wikitext103/wikitext103.idx
    {output-dir}/lambada/lambada.bin
    {output-dir}/lambada/lambada.idx

Usage:
    python scripts/download_benchmarks.py \
        --output-dir /path/to/benchmarks/neox \
        --tokenizer neox \
        --cluster juwels \
        --datasets wikitext2 wikitext103 lambada
"""

import argparse
import logging
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import numpy

# Add project root to path for titan_oellm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from titan_oellm.cluster_config import get_tokenizer_path
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import SciHFTokenizer

logger = logging.getLogger(__name__)

# Index file format constants
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
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    hf_name: str  # HuggingFace dataset name
    hf_config: Optional[str]  # HuggingFace dataset config/subset
    split: str  # Dataset split to use
    text_column: str  # Column containing text
    output_name: str  # Output file prefix


# Dataset configurations
DATASET_CONFIGS = {
    "wikitext2": DatasetConfig(
        hf_name="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
        split="test",
        text_column="text",
        output_name="wikitext2",
    ),
    "wikitext103": DatasetConfig(
        hf_name="Salesforce/wikitext",
        hf_config="wikitext-103-raw-v1",
        split="test",
        text_column="text",
        output_name="wikitext103",
    ),
    "lambada": DatasetConfig(
        hf_name="EleutherAI/lambada_openai",
        hf_config="default",
        split="test",
        text_column="text",
        output_name="lambada",
    ),
}


def get_tokenizer(tokenizer_name: str, cluster: str, user: str = "joerg") -> SciHFTokenizer:
    """Get tokenizer from cluster_paths.toml.

    Args:
        tokenizer_name: Tokenizer name from cluster_paths.toml (e.g., neox, nemotron)
        cluster: Cluster name (juwels, jupiter, capella)
        user: User directory for cluster_paths.toml (default: joerg)

    Returns:
        SciHFTokenizer instance
    """
    tokenizer_path = get_tokenizer_path(
        tokenizer=tokenizer_name,
        cluster=cluster,
        user=user
    )

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    return SciHFTokenizer(tokenizer_path)


def write_mmap_dataset(
    output_prefix: str,
    documents: List[np.ndarray],
    dtype: Type[numpy.number] = numpy.uint16,
) -> Dict[str, int]:
    """Write documents to bin/idx format compatible with MMapDataset.

    Args:
        output_prefix: Path prefix for output files (.bin and .idx will be added)
        documents: List of tokenized documents as numpy arrays
        dtype: Data type for tokens

    Returns:
        Dictionary with statistics about the written dataset
    """
    bin_path = f"{output_prefix}.bin"
    idx_path = f"{output_prefix}.idx"

    sequence_lengths = []
    sequence_pointers = []
    current_pointer = 0

    # Write binary data
    with open(bin_path, 'wb') as bin_file:
        for doc in documents:
            doc_array = np.array(doc, dtype=dtype)
            doc_array.tofile(bin_file)

            sequence_lengths.append(len(doc))
            sequence_pointers.append(current_pointer)
            current_pointer += len(doc) * np.dtype(dtype).itemsize

    # Write index file
    sequence_count = len(sequence_lengths)

    with open(idx_path, 'wb') as idx_file:
        # Write header
        idx_file.write(_INDEX_HEADER)

        # Write version
        idx_file.write(struct.pack("<Q", 1))

        # Write dtype code
        dtype_code = DType.code_from_dtype(dtype)
        idx_file.write(struct.pack("<B", dtype_code))

        # Write counts (sequence_count, document_count - same for us)
        idx_file.write(struct.pack("<Q", sequence_count))
        idx_file.write(struct.pack("<Q", sequence_count))

        # Write sequence lengths
        np.array(sequence_lengths, dtype=np.int32).tofile(idx_file)

        # Write sequence pointers
        np.array(sequence_pointers, dtype=np.int64).tofile(idx_file)

        # Write document indices (1:1 mapping)
        np.arange(sequence_count, dtype=np.int64).tofile(idx_file)

    total_tokens = sum(sequence_lengths)
    return {
        'documents': sequence_count,
        'total_tokens': total_tokens,
        'avg_doc_length': total_tokens / sequence_count if sequence_count > 0 else 0,
        'bin_size_mb': current_pointer / (1024 * 1024),
    }


def download_and_process_dataset(
    config: DatasetConfig,
    tokenizer,
    output_dir: Path,
    min_doc_length: int = 1,
) -> Dict[str, int]:
    """Download and process a single benchmark dataset.

    Args:
        config: Dataset configuration
        tokenizer: Tokenizer to use
        output_dir: Output directory
        min_doc_length: Minimum document length in tokens (filter out shorter)

    Returns:
        Statistics about the processed dataset
    """
    from datasets import load_dataset

    logger.info(f"Downloading {config.hf_name} ({config.hf_config})...")

    # Load dataset
    if config.hf_config:
        dataset = load_dataset(config.hf_name, config.hf_config, split=config.split)
    else:
        dataset = load_dataset(config.hf_name, split=config.split)

    logger.info(f"Processing {len(dataset)} examples...")

    # Determine optimal dtype based on vocabulary size
    vocab_size = tokenizer.vocab_size
    dtype = DType.optimal_dtype(vocab_size)
    logger.info(f"Using dtype: {dtype.__name__} (vocab_size: {vocab_size})")

    # Tokenize documents
    documents = []
    skipped = 0

    for i, example in enumerate(dataset):
        text = example[config.text_column]

        # Skip empty texts
        if not text or not text.strip():
            skipped += 1
            continue

        # Tokenize (no BOS/EOS for benchmark evaluation)
        tokens = tokenizer.encode(text, bos=False, eos=False)

        # Skip too short documents
        if len(tokens) < min_doc_length:
            skipped += 1
            continue

        documents.append(np.array(tokens, dtype=dtype))

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{len(dataset)} examples")

    logger.info(f"Tokenized {len(documents)} documents (skipped {skipped})")

    # Create subfolder for this dataset
    dataset_dir = output_dir / config.output_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write to bin/idx format in subfolder
    output_prefix = str(dataset_dir / config.output_name)
    stats = write_mmap_dataset(output_prefix, documents, dtype=dtype)

    logger.info(f"Written {config.output_name}: {stats['documents']} docs, "
                f"{stats['total_tokens']:,} tokens, {stats['bin_size_mb']:.2f} MB")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess benchmark datasets for LLM evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name from cluster_paths.toml (e.g., neox, nemotron)"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        help="Cluster name: juwels, jupiter, capella"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="joerg",
        help="User directory for cluster_paths.toml (default: joerg)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "wikitext103", "lambada"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to download and process"
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=1,
        help="Minimum document length in tokens (default: 1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get tokenizer from cluster_paths.toml
    logger.info(f"Loading tokenizer '{args.tokenizer}' for cluster '{args.cluster}'")
    tokenizer = get_tokenizer(args.tokenizer, args.cluster, args.user)

    # Process each dataset
    all_stats = {}
    for dataset_name in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'='*60}")

        config = DATASET_CONFIGS[dataset_name]
        stats = download_and_process_dataset(
            config=config,
            tokenizer=tokenizer,
            output_dir=output_dir,
            min_doc_length=args.min_doc_length,
        )
        all_stats[dataset_name] = stats

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Summary")
    logger.info(f"{'='*60}")
    for name, stats in all_stats.items():
        logger.info(f"{name}:")
        logger.info(f"  Documents: {stats['documents']:,}")
        logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        logger.info(f"  Avg length: {stats['avg_doc_length']:.1f}")
        logger.info(f"  Size: {stats['bin_size_mb']:.2f} MB")

    logger.info(f"\nOutput files written to: {output_dir}")
    logger.info("Files can be used with MMapDataset for evaluation.")


if __name__ == "__main__":
    main()
