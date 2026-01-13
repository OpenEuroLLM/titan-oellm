"""Convert HuggingFace dataset to MMap format (bin/idx) for SFT training.

This script converts HuggingFace datasets to the MMap format used by titan-sci's
MMapDataset for efficient random-access training.

Usage:
    # Convert Alpaca dataset
    python convert_hf_to_mmap.py \
        --dataset tatsu-lab/alpaca \
        --output ./data/alpaca_sft \
        --tokenizer ./assets/hf/Qwen3-0.6B \
        --text_column text

    # Convert local dataset
    python convert_hf_to_mmap.py \
        --dataset ./my_local_dataset \
        --output ./data/my_sft \
        --tokenizer ./assets/hf/Qwen3-8B
"""
import argparse
import struct
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class MMapDatasetWriter:
    """Write tokenized documents to MMap format (.bin + .idx files).

    MMap format (compatible with titan_oellm.datasets.dataloader.mmap_dataset):
    - .idx file: Header + metadata + sequence_lengths (int32) + sequence_pointers (int64) + document_indices (int64)
    - .bin file: Raw tokenized data (uint16 for vocab < 65500, else int32)
    """

    def __init__(self, output_prefix: str, vocab_size: int = 151936):
        """Initialize the writer.

        Args:
            output_prefix: Path prefix for output files (creates {prefix}.bin and {prefix}.idx)
            vocab_size: Vocabulary size to determine optimal dtype (uint16 for small vocabs)
        """
        self.output_prefix = output_prefix

        # Use uint16 for small vocabs (saves space), int32 otherwise
        self.dtype = np.uint16 if vocab_size < 65500 else np.int32
        self.dtype_code = 8 if vocab_size < 65500 else 4  # uint16=8, int32=4

        # Ensure output directory exists
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        self.bin_file = open(f"{output_prefix}.bin", "wb")
        self.sequence_lengths = []
        self.sequence_pointers = []
        self.document_indices = []
        self.current_offset = 0

    def add_document(self, tokens: list[int]) -> None:
        """Add a tokenized document.

        Args:
            tokens: List of token IDs
        """
        if len(tokens) == 0:
            return

        # Record document boundary
        self.document_indices.append(len(self.sequence_lengths))

        # Write tokens to bin file
        arr = np.array(tokens, dtype=self.dtype)
        self.bin_file.write(arr.tobytes())

        # Record metadata
        self.sequence_lengths.append(len(tokens))
        self.sequence_pointers.append(self.current_offset)
        self.current_offset += arr.nbytes

    def finalize(self) -> None:
        """Write index file and close all files."""
        self.bin_file.close()

        # Write index file
        with open(f"{self.output_prefix}.idx", "wb") as f:
            # Header
            f.write(_INDEX_HEADER)
            f.write(struct.pack("<Q", 1))  # version
            f.write(struct.pack("<B", self.dtype_code))  # dtype code
            f.write(struct.pack("<Q", len(self.sequence_lengths)))  # sequence count
            f.write(struct.pack("<Q", len(self.document_indices)))  # document count

            # Arrays
            np.array(self.sequence_lengths, dtype=np.int32).tofile(f)
            np.array(self.sequence_pointers, dtype=np.int64).tofile(f)
            np.array(self.document_indices, dtype=np.int64).tofile(f)

        print(f"Created {self.output_prefix}.bin ({self.current_offset / 1e9:.2f} GB)")
        print(f"Created {self.output_prefix}.idx ({len(self.sequence_lengths)} sequences, {len(self.document_indices)} documents)")


def convert_hf_to_mmap(
    dataset_name: str,
    output_prefix: str,
    tokenizer_path: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: int = -1,
    add_bos: bool = True,
    add_eos: bool = True,
):
    """Convert HuggingFace dataset to MMap format.

    Args:
        dataset_name: HF dataset name (e.g., "tatsu-lab/alpaca") or local path
        output_prefix: Output path prefix (creates {prefix}.bin and {prefix}.idx)
        tokenizer_path: Path to tokenizer (e.g., "./assets/hf/Qwen3-0.6B")
        split: Dataset split to use
        text_column: Column containing text to tokenize
        max_samples: Max samples to convert (-1 = all)
        add_bos: Add BOS token at the start of each document
        add_eos: Add EOS token at the end of each document
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading dataset {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Get vocab size for optimal dtype selection
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size} -> using {'uint16' if vocab_size < 65500 else 'int32'}")

    writer = MMapDatasetWriter(output_prefix, vocab_size)

    total_tokens = 0
    for sample in tqdm(dataset, desc="Converting"):
        text = sample[text_column]
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if add_bos and tokenizer.bos_token_id is not None:
            tokens = [tokenizer.bos_token_id] + tokens
        if add_eos and tokenizer.eos_token_id is not None:
            tokens = tokens + [tokenizer.eos_token_id]

        writer.add_document(tokens)
        total_tokens += len(tokens)

    writer.finalize()
    print(f"Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to MMap format for titan-sci training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert Alpaca dataset with Qwen3 tokenizer
    python convert_hf_to_mmap.py \\
        --dataset tatsu-lab/alpaca \\
        --output ./data/alpaca_sft \\
        --tokenizer ./assets/hf/Qwen3-0.6B \\
        --text_column text

    # Convert local JSONL dataset
    python convert_hf_to_mmap.py \\
        --dataset json \\
        --data_files ./my_data.jsonl \\
        --output ./data/my_sft \\
        --tokenizer Qwen/Qwen3-8B
        """,
    )
    parser.add_argument("--dataset", required=True, help="HF dataset name or local path")
    parser.add_argument("--output", required=True, help="Output path prefix (creates .bin and .idx)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path (local or HF hub)")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--text_column", default="text", help="Column containing text (default: text)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to convert (-1 = all)")
    parser.add_argument("--no_bos", action="store_true", help="Don't add BOS token")
    parser.add_argument("--no_eos", action="store_true", help="Don't add EOS token")
    args = parser.parse_args()

    convert_hf_to_mmap(
        dataset_name=args.dataset,
        output_prefix=args.output,
        tokenizer_path=args.tokenizer,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
    )
