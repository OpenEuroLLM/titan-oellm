import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from titan_oellm.constants import IGNORE_INDEX


def collate_function_document_eval(
    batch: List[Dict],
    seq_len: int,
    ignore_index: int = IGNORE_INDEX,
    pad_id: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collate function for document-level evaluation (no concatenation/sequencing).

    Each document is independently padded or truncated to seq_len.
    This enables proper per-document perplexity evaluation without cross-document dependencies.

    Args:
        batch: List of samples from dataset. Each sample has 'tokens' as np.ndarray.
        seq_len: Target sequence length. Documents are padded/truncated to this length.
        ignore_index: Index to mask padding positions in target sequence (default: IGNORE_INDEX).
        pad_id: Token ID to use for padding (default: 0).

    Returns:
        Tuple of (input_dict, target_tensor):
        - input_dict: {"input": tensor of shape [batch_size, seq_len]}
        - target_tensor: tensor of shape [batch_size, seq_len] with padded positions masked as ignore_index
    """
    src_sequences = []
    trg_sequences = []

    for sample in batch:
        tokens = sample['tokens']
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens).long()
        elif not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)

        doc_len = len(tokens)

        if doc_len >= seq_len + 1:
            # Truncate: take first seq_len+1 tokens for src[:-1]/trg[1:]
            src = tokens[:seq_len]
            trg = tokens[1:seq_len + 1]
        else:
            # Pad: document is shorter than seq_len
            # src = tokens[:-1] padded to seq_len
            # trg = tokens[1:] padded to seq_len with padding masked
            src = tokens[:-1] if doc_len > 1 else tokens
            trg = tokens[1:] if doc_len > 1 else tokens

            src_pad_len = seq_len - len(src)
            trg_pad_len = seq_len - len(trg)

            if src_pad_len > 0:
                src = F.pad(src, (0, src_pad_len), value=pad_id)
            if trg_pad_len > 0:
                # Create mask for non-padded positions
                trg_mask = torch.ones(len(trg), dtype=torch.bool)
                trg = F.pad(trg, (0, trg_pad_len), value=ignore_index)
                trg_mask = F.pad(trg_mask, (0, trg_pad_len), value=False)
                # Mask padded positions
                trg = trg.masked_fill(~trg_mask, ignore_index)

        src_sequences.append(src)
        trg_sequences.append(trg)

    # Stack into batch tensors
    src_batch = torch.stack(src_sequences, dim=0)  # [batch_size, seq_len]
    trg_batch = torch.stack(trg_sequences, dim=0)  # [batch_size, seq_len]

    return {"input": src_batch}, trg_batch


def collate_function(batch, ignore_index, use_flash_attention=False, eos_id=None, seq_len=None, max_cu_seqlens_size=None):
    """Collate function for batching samples.

    Args:
        batch: List of samples to collate.
        ignore_index: Index to use for padding target sequences (typically -100).
        use_flash_attention: If True, return cu_seqlens/max_seqlen for Flash Attention.
                            If False (default), return simple input dict for standard attention.
        eos_id: EOS token ID for detecting truncated documents (used with flash attention).
        seq_len: Fixed sequence length from config. Required for flash attention to ensure
                static max_seqlen for torch.compile fullgraph mode.
        max_cu_seqlens_size: Fixed size for cu_seqlens tensor to avoid torch.compile recompilations.
                            Should be: batch_size * (seq_len // min_doc_len + 1) + 1
    """
    has_mask = "mask" in batch[0]
    has_metadata = "set_name" in batch[0]

    src_seq = []
    trg_seq = []
    batch_seqlens = []
    sample_seqlens = []
    sample_length = []

    sample_mask = [] if has_mask else None
    set_names = [] if has_metadata else None
    set_epochs = [] if has_metadata else None

    # Track truncated samples for cross-sample merging (only for flash attention)
    truncated_samples = [] if use_flash_attention and eos_id is not None else None

    for i, sample in enumerate(batch):

        srq_sample = [s[:-1] for s in sample['tokens']]
        trg_sample = [s[1:] for s in sample['tokens']]
        seqlens = list(sample['seqlen'])  # copy to avoid mutating original
        seqlens[-1] -= 1

        # Check if this sample ends with truncated doc (last token != eos)
        if truncated_samples is not None:
            tokens = sample['tokens'][0]
            # tokens[-1] is the last token in the sequence
            # If it's not EOS, the last document was truncated
            if tokens[-1] != eos_id:
                truncated_samples.append(i)

        batch_seqlens.extend(seqlens)
        sample_seqlens.append(seqlens)
        sample_length.append(sum(seqlens))

        src_seq.append(torch.from_numpy(np.concatenate(srq_sample)).long())
        trg_sample = torch.from_numpy(np.concatenate(trg_sample)).long()

        if has_mask:
            mask = torch.from_numpy(np.concatenate([s[1:] for s in sample['mask']])).long()
            trg_sample = trg_sample.masked_fill(mask == 0, ignore_index)
            sample_mask.append(mask)

        trg_seq.append(trg_sample)

        if has_metadata:
            set_names.append(sample["set_name"])
            set_epochs.append(sample["set_epoch"])

    # Use fixed seq_len from config for flash attention (enables fullgraph compilation)
    # Fall back to computed max for non-flash attention mode
    if seq_len is None:
        seq_len = max(sample_length)

    # Merge truncated doc boundaries across consecutive samples (for flash attention)
    # If sample N ends with truncated doc, merge its last seqlen with sample N+1's first seqlen
    if truncated_samples:
        # Build cumulative index mapping: sample_idx -> start index in batch_seqlens
        sample_start_indices = []
        idx = 0
        for seqlens in sample_seqlens:
            sample_start_indices.append(idx)
            idx += len(seqlens)

        # Process truncated samples in reverse order to preserve indices
        for sample_idx in reversed(truncated_samples):
            # Only merge if there's a next sample
            if sample_idx + 1 < len(sample_seqlens):
                # Index of last seqlen entry for truncated sample
                last_idx = sample_start_indices[sample_idx] + len(sample_seqlens[sample_idx]) - 1
                # Index of first seqlen entry for next sample
                next_first_idx = sample_start_indices[sample_idx + 1]
                # Only merge if combined length doesn't exceed 2x sequence length
                # (merged docs can span two samples, so up to 2*seq_len)
                merged_len = batch_seqlens[last_idx] + batch_seqlens[next_first_idx]
                if merged_len > 2 * seq_len:
                    continue
                # Merge: add next sample's first seqlen to truncated sample's last seqlen
                batch_seqlens[last_idx] = merged_len
                # Remove the merged entry
                batch_seqlens.pop(next_first_idx)
                # Update start indices for subsequent samples
                for j in range(sample_idx + 1, len(sample_start_indices)):
                    sample_start_indices[j] -= 1

    src_seq = torch.nn.utils.rnn.pad_sequence(src_seq, batch_first=True, padding_value=0)
    trg_seq = torch.nn.utils.rnn.pad_sequence(trg_seq, batch_first=True, padding_value=ignore_index)


    if use_flash_attention:
        batch_seqlens = torch.tensor(batch_seqlens, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(batch_seqlens, dim=0, dtype=torch.int32), (1, 0))
        num_docs = cu_seqlens.shape[0]

        # Pad cu_seqlens to fixed size for torch.compile fullgraph mode
        # Padding with the last value (total tokens) creates zero-length docs at end
        if max_cu_seqlens_size is not None and num_docs < max_cu_seqlens_size:
            pad_size = max_cu_seqlens_size - num_docs
            cu_seqlens = F.pad(cu_seqlens, (0, pad_size), value=cu_seqlens[-1].item())

        # Use 2*seq_len for max_seqlen to enable fullgraph compilation
        # (merged docs across samples can span up to 2*seq_len tokens)
        max_seqlen = 2 * seq_len

        return {
            "input": src_seq,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }, trg_seq
    else:
        # Standard output for SDPA/FlexAttention
        return {"input": src_seq}, trg_seq


class Collator():
    def __init__(self, sample_loader, device, batch_size: int, buffer_size: int=3,
                 ignore_index: int = IGNORE_INDEX, use_flash_attention: bool = False, eos_id: int = None):
        self.sample_loader = iter(sample_loader)
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.ignore_index = ignore_index
        self.use_flash_attention = use_flash_attention
        self.eos_id = eos_id

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                sample = next(self.sample_loader)
                batch.append(sample)
            batch = collate_function(
                batch, self.ignore_index,
                use_flash_attention=self.use_flash_attention,
                eos_id=self.eos_id
            )
            return batch
        except StopIteration:
            raise StopIteration
