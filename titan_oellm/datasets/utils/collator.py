import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from titan_oellm.constants import IGNORE_INDEX

# Module-level accumulator for packing statistics, read by EnhancedMetricsProcessor.
# Safe because training is single-threaded per rank.
_last_packing_stats: dict[str, float] = {}


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
    validation_generation_starts: list[int] = []
    has_generation_start = False

    for sample in batch:
        tokens = sample["tokens"]
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens).long()
        elif not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)

        doc_len = len(tokens)

        if doc_len >= seq_len + 1:
            src = tokens[:seq_len]
            trg_unmasked = tokens[1 : seq_len + 1]
        else:
            src = tokens[:-1] if doc_len > 1 else tokens
            trg_unmasked = tokens[1:] if doc_len > 1 else tokens

            src_pad_len = seq_len - len(src)
            trg_pad_len = seq_len - len(trg_unmasked)

            if src_pad_len > 0:
                src = F.pad(src, (0, src_pad_len), value=pad_id)
            if trg_pad_len > 0:
                trg_unmasked = F.pad(trg_unmasked, (0, trg_pad_len), value=ignore_index)

        trg = trg_unmasked.clone()

        mask_source = sample.get("mask", sample.get("loss_mask"))
        if mask_source is not None:
            mask_tensor = torch.tensor(mask_source, dtype=torch.long)
            if mask_tensor.numel() > 0:
                mask_targets = mask_tensor[1 : 1 + len(trg_unmasked)]
            else:
                mask_targets = torch.zeros(len(trg_unmasked), dtype=torch.long)
            if len(mask_targets) < len(trg_unmasked):
                mask_targets = F.pad(mask_targets, (0, len(trg_unmasked) - len(mask_targets)), value=0)
            trg = trg.masked_fill(mask_targets == 0, ignore_index)

        if "generation_start" in sample:
            has_generation_start = True
            generation_start = sample.get("generation_start", 0)
            if isinstance(generation_start, torch.Tensor):
                generation_start = generation_start.item()
            validation_generation_starts.append(int(generation_start))

        src_sequences.append(src)
        trg_sequences.append(trg)

    # Stack into batch tensors
    src_batch = torch.stack(src_sequences, dim=0)  # [batch_size, seq_len]
    trg_batch = torch.stack(trg_sequences, dim=0)  # [batch_size, seq_len]

    input_dict = {"input": src_batch}
    if has_generation_start:
        input_dict["generation_start"] = torch.tensor(validation_generation_starts, dtype=torch.long)
    return input_dict, trg_batch


def collate_synth_function_document_eval(
    batch: List[Dict],
    seq_len: int,
    ignore_index: int = IGNORE_INDEX,
    pad_id: int = 0,
    include_loss_mask: bool = False,
    include_unmasked_labels: bool = False,
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
    loss_masks = [] if include_loss_mask else None
    unmasked_labels = [] if include_unmasked_labels else None
    validation_generation_starts: list[int] = []
    has_generation_start = False

    for sample in batch:
        tokens = sample["tokens"]
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens).long()
        elif not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)

        doc_len = len(tokens)

        if doc_len >= seq_len + 1:
            src = tokens[:seq_len]
            trg_unmasked = tokens[1 : seq_len + 1]
        else:
            src = tokens[:-1] if doc_len > 1 else tokens
            trg_unmasked = tokens[1:] if doc_len > 1 else tokens

            src_pad_len = seq_len - len(src)
            trg_pad_len = seq_len - len(trg_unmasked)

            if src_pad_len > 0:
                src = F.pad(src, (0, src_pad_len), value=pad_id)
            if trg_pad_len > 0:
                trg_unmasked = F.pad(trg_unmasked, (0, trg_pad_len), value=ignore_index)

        trg = trg_unmasked.clone()

        mask_source = sample.get("mask", sample.get("loss_mask"))
        if mask_source is not None:
            mask_tensor = torch.tensor(mask_source, dtype=torch.long)
            if mask_tensor.numel() > 0:
                mask_targets = mask_tensor[1 : 1 + len(trg_unmasked)]
            else:
                mask_targets = torch.zeros(len(trg_unmasked), dtype=torch.long)
            if len(mask_targets) < len(trg_unmasked):
                mask_targets = F.pad(mask_targets, (0, len(trg_unmasked) - len(mask_targets)), value=0)
            trg = trg.masked_fill(mask_targets == 0, ignore_index)
            if loss_masks is not None:
                loss_masks.append(mask_targets)
        elif loss_masks is not None:
            loss_masks.append(torch.ones(len(trg_unmasked), dtype=torch.long))

        if unmasked_labels is not None:
            unmasked_labels.append(trg_unmasked)

        if "generation_start" in sample:
            has_generation_start = True
            generation_start = sample.get("generation_start", 0)
            if isinstance(generation_start, torch.Tensor):
                generation_start = generation_start.item()
            validation_generation_starts.append(int(generation_start))

        src_sequences.append(src)
        trg_sequences.append(trg)

    # Stack into batch tensors
    src_batch = torch.stack(src_sequences, dim=0)  # [batch_size, seq_len]
    trg_batch = torch.stack(trg_sequences, dim=0)  # [batch_size, seq_len]

    input_dict = {"input": src_batch}
    if loss_masks is not None:
        input_dict["loss_mask"] = torch.stack(loss_masks, dim=0)
    if unmasked_labels is not None:
        input_dict["labels_unmasked"] = torch.stack(unmasked_labels, dim=0)
    if has_generation_start:
        input_dict["generation_start"] = torch.tensor(validation_generation_starts, dtype=torch.long)
    return input_dict, trg_batch


def collate_function(
    batch,
    ignore_index,
    use_flash_attention=False,
    use_document_mask=False,
    eos_id=None,
    seq_len=None,
    max_cu_seqlens_size=None,
    mask_eot_loss=False,
):
    """Collate function for batching samples.

    Args:
        batch: List of samples to collate.
        ignore_index: Index to use for padding target sequences (typically -100).
        use_flash_attention: If True, return cu_seqlens/max_seqlen for Flash Attention.
                            If False (default), return simple input dict for standard attention.
        use_document_mask: If True, return a [B, 1, S, S] boolean attention mask for SDPA
                          that combines causal masking with document boundary isolation.
                          Built from per-sample seqlens (no EOS detection needed).
        eos_id: EOS token ID for detecting truncated documents (used with flash attention).
        seq_len: Fixed sequence length from config. Required for flash attention to ensure
                static max_seqlen for torch.compile fullgraph mode.
        max_cu_seqlens_size: Fixed size for cu_seqlens tensor to avoid torch.compile recompilations.
                            Should be: batch_size * (seq_len // min_doc_len + 1) + 1
    """
    has_mask = "mask" in batch[0] or "loss_mask" in batch[0]
    has_metadata = "set_name" in batch[0]

    src_seq = []
    trg_seq = []
    batch_seqlens = []
    sample_seqlens = []
    sample_length = []

    sample_mask = [] if has_mask else None
    set_names = [] if has_metadata else None
    set_epochs = [] if has_metadata else None

    # Track truncated samples for metrics and cross-sample merging
    truncated_samples = [] if eos_id is not None else None

    for i, sample in enumerate(batch):
        srq_sample = [s[:-1] for s in sample["tokens"]]
        trg_sample = [s[1:] for s in sample["tokens"]]
        seqlens = list(sample["seqlen"])  # copy to avoid mutating original
        seqlens[-1] -= 1

        # Check if this sample ends with truncated doc (last token != eos)
        if truncated_samples is not None:
            tokens = sample["tokens"][0]
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
            mask_source = sample.get("mask", sample.get("loss_mask"))
            mask = torch.from_numpy(np.concatenate([s[1:] for s in mask_source])).long()
            trg_sample = trg_sample.masked_fill(mask == 0, ignore_index)
            sample_mask.append(mask)

        if mask_eot_loss and eos_id is not None:
            trg_sample = trg_sample.masked_fill(trg_sample == eos_id, ignore_index)

        trg_seq.append(trg_sample)

        if has_metadata:
            set_names.append(sample["set_name"])
            set_epochs.append(sample["set_epoch"])

    # Use fixed seq_len from config for flash attention (enables fullgraph compilation)
    # Fall back to computed max for non-flash attention mode
    if seq_len is None:
        seq_len = max(sample_length)

    # Compute packing statistics for logging
    global _last_packing_stats
    batch_size_actual = len(batch)
    total_real_tokens = sum(sample_length)
    total_padded_tokens = batch_size_actual * seq_len - total_real_tokens
    total_doc_fragments = sum(len(sl) for sl in sample_seqlens)
    avg_docs_per_seq = total_doc_fragments / batch_size_actual if batch_size_actual > 0 else 0.0
    real_tokens_pct = total_real_tokens / (batch_size_actual * seq_len) * 100 if batch_size_actual > 0 else 0.0
    truncated_seqs = float(len(truncated_samples)) if truncated_samples is not None else 0.0
    truncation_rate = truncated_seqs / batch_size_actual * 100 if batch_size_actual > 0 else 0.0
    _last_packing_stats = {
        "dataset/real_tokens": float(total_real_tokens),
        "dataset/padding_tokens": float(total_padded_tokens),
        "dataset/real_tokens_pct": real_tokens_pct,
        "dataset/avg_docs_per_seq": avg_docs_per_seq,
        "dataset/batch_size": float(batch_size_actual),
        "dataset/truncated_seqs": truncated_seqs,
        "dataset/truncation_rate_pct": truncation_rate,
    }

    # Merge truncated doc boundaries across consecutive samples (only for flash attention cu_seqlens)
    # If sample N ends with truncated doc, merge its last seqlen with sample N+1's first seqlen
    if use_flash_attention and truncated_samples:
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

        input_dict = {
            "input": src_seq,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        return input_dict, trg_seq
    else:
        # Standard output for SDPA/FlexAttention
        result = {"input": src_seq}

        if use_document_mask:
            # Build [B, 1, S, S] boolean attention mask from document seqlens.
            # Combines causal masking with document boundary isolation for SDPA.
            padded_seq_len = src_seq.shape[1]
            causal = torch.ones(padded_seq_len, padded_seq_len, dtype=torch.bool).tril()
            doc_masks = []
            for seqlens in sample_seqlens:
                doc_ids = torch.zeros(padded_seq_len, dtype=torch.int32)
                pos = 0
                for doc_idx, doc_len in enumerate(seqlens):
                    end = min(pos + doc_len, padded_seq_len)
                    doc_ids[pos:end] = doc_idx
                    pos = end
                # Padding positions (pos..padded_seq_len) keep doc_id=0 but are
                # already masked by causal mask (padding is at end of shorter seqs)
                doc_masks.append(doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))
            doc_mask = torch.stack(doc_masks).unsqueeze(1)  # [B, 1, S, S]
            result["attention_masks"] = causal.unsqueeze(0).unsqueeze(0) & doc_mask

        return result, trg_seq


def collate_synth_function(
    batch,
    ignore_index,
    use_flash_attention=False,
    use_document_mask=False,
    eos_id=None,
    seq_len=None,
    max_cu_seqlens_size=None,
    include_loss_mask=False,
    include_unmasked_labels=False,
):
    """Collate function for batching samples.

    Args:
        batch: List of samples to collate.
        ignore_index: Index to use for padding target sequences (typically -100).
        use_flash_attention: If True, return cu_seqlens/max_seqlen for Flash Attention.
                            If False (default), return simple input dict for standard attention.
        use_document_mask: If True, return a [B, 1, S, S] boolean attention mask for SDPA
                          that combines causal masking with document boundary isolation.
                          Built from per-sample seqlens (no EOS detection needed).
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
    sample_unmasked = [] if has_mask and include_unmasked_labels else None
    set_names = [] if has_metadata else None
    set_epochs = [] if has_metadata else None

    # Track truncated samples for metrics and cross-sample merging
    truncated_samples = [] if eos_id is not None else None

    for i, sample in enumerate(batch):
        srq_sample = [s[:-1] for s in sample["tokens"]]
        trg_sample = [s[1:] for s in sample["tokens"]]
        seqlens = list(sample["seqlen"])  # copy to avoid mutating original
        seqlens[-1] -= 1

        # Check if this sample ends with truncated doc (last token != eos)
        if truncated_samples is not None:
            tokens = sample["tokens"][0]
            # tokens[-1] is the last token in the sequence
            # If it's not EOS, the last document was truncated
            if tokens[-1] != eos_id:
                truncated_samples.append(i)

        batch_seqlens.extend(seqlens)
        sample_seqlens.append(seqlens)
        sample_length.append(sum(seqlens))

        src_seq.append(torch.from_numpy(np.concatenate(srq_sample)).long())
        trg_sample = torch.from_numpy(np.concatenate(trg_sample)).long()

        if sample_unmasked is not None:
            sample_unmasked.append(trg_sample.clone())

        if has_mask:
            mask_source = sample.get("mask", sample.get("loss_mask"))
            mask = torch.from_numpy(np.concatenate([s[1:] for s in mask_source])).long()
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

    # Compute packing statistics for logging
    global _last_packing_stats
    batch_size_actual = len(batch)
    total_real_tokens = sum(sample_length)
    total_padded_tokens = batch_size_actual * seq_len - total_real_tokens
    total_doc_fragments = sum(len(sl) for sl in sample_seqlens)
    avg_docs_per_seq = total_doc_fragments / batch_size_actual if batch_size_actual > 0 else 0.0
    real_tokens_pct = total_real_tokens / (batch_size_actual * seq_len) * 100 if batch_size_actual > 0 else 0.0
    truncated_seqs = float(len(truncated_samples)) if truncated_samples is not None else 0.0
    truncation_rate = truncated_seqs / batch_size_actual * 100 if batch_size_actual > 0 else 0.0
    _last_packing_stats = {
        "dataset/real_tokens": float(total_real_tokens),
        "dataset/padding_tokens": float(total_padded_tokens),
        "dataset/real_tokens_pct": real_tokens_pct,
        "dataset/avg_docs_per_seq": avg_docs_per_seq,
        "dataset/batch_size": float(batch_size_actual),
        "dataset/truncated_seqs": truncated_seqs,
        "dataset/truncation_rate_pct": truncation_rate,
    }

    # Merge truncated doc boundaries across consecutive samples (only for flash attention cu_seqlens)
    # If sample N ends with truncated doc, merge its last seqlen with sample N+1's first seqlen
    if use_flash_attention and truncated_samples:
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

        input_dict = {
            "input": src_seq,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        if include_loss_mask and has_mask:
            loss_mask = torch.nn.utils.rnn.pad_sequence(sample_mask, batch_first=True, padding_value=0)
            input_dict["loss_mask"] = loss_mask
        if sample_unmasked is not None:
            unmasked_labels = torch.nn.utils.rnn.pad_sequence(
                sample_unmasked, batch_first=True, padding_value=ignore_index
            )
            input_dict["labels_unmasked"] = unmasked_labels
        return input_dict, trg_seq
    else:
        # Standard output for SDPA/FlexAttention
        result = {"input": src_seq}

        if use_document_mask:
            # Build [B, 1, S, S] boolean attention mask from document seqlens.
            # Combines causal masking with document boundary isolation for SDPA.
            padded_seq_len = src_seq.shape[1]
            causal = torch.ones(padded_seq_len, padded_seq_len, dtype=torch.bool).tril()
            doc_masks = []
            for seqlens in sample_seqlens:
                doc_ids = torch.zeros(padded_seq_len, dtype=torch.int32)
                pos = 0
                for doc_idx, doc_len in enumerate(seqlens):
                    end = min(pos + doc_len, padded_seq_len)
                    doc_ids[pos:end] = doc_idx
                    pos = end
                # Padding positions (pos..padded_seq_len) keep doc_id=0 but are
                # already masked by causal mask (padding is at end of shorter seqs)
                doc_masks.append(doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))
            doc_mask = torch.stack(doc_masks).unsqueeze(1)  # [B, 1, S, S]
            result["attention_masks"] = causal.unsqueeze(0).unsqueeze(0) & doc_mask

        return result, trg_seq


class Collator:
    def __init__(
        self,
        sample_loader,
        device,
        batch_size: int,
        buffer_size: int = 3,
        ignore_index: int = IGNORE_INDEX,
        use_flash_attention: bool = False,
        eos_id: int = None,
    ):
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
                batch, self.ignore_index, use_flash_attention=self.use_flash_attention, eos_id=self.eos_id
            )
            return batch
        except StopIteration:
            raise StopIteration
