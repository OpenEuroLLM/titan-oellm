import numpy as np
from typing import Iterator, Optional, List, Dict, Any
from torch.utils.data import IterableDataset
import torch

class StreamingSequencer(IterableDataset):
    """
    Alternative streaming-based sequencer for maximum performance.
    Processes data in a streaming fashion without complex buffering.

    Args:
        dataset: Source dataset yielding document samples
        sequence_length: Target sequence length for output samples
        min_sequence_length: Minimum document length to include (default: 0)
        drop_last: Whether to drop incomplete final sequence (default: False)
        eos_id: EOS token ID to insert between documents for document masking (default: None = no separator)
    """

    def __init__(
            self,
            dataset: IterableDataset,
            sequence_length: int,
            min_sequence_length: int = 0,
            drop_last: bool = False,
            eos_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.drop_last = drop_last
        self.eos_id = eos_id

        # Streaming buffer
        self.token_stream = []
        self.mask_stream = [] if hasattr(self, '_detect_masks') else None

        self.dataset_iter = iter(self.dataset) # TODO test

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Simple streaming approach"""


        # Initialize streaming buffers
        token_stream = []
        seqlen_stream = []
        mask_stream = []
        has_masks = False

        for sample in self.dataset_iter:
            # Add sample tokens to stream
            sample_tokens = sample['tokens']

            if len(sample_tokens) < self.min_sequence_length:
                continue

            token_stream.extend(sample_tokens)
            seqlen_stream.append(len(sample_tokens))

            # Add EOS token between documents for document masking (FlexAttention)
            if self.eos_id is not None:
                token_stream.append(self.eos_id)
                seqlen_stream[-1] += 1

            # Handle masks if present
            if 'mask' in sample:
                if not has_masks:
                    has_masks = True
                mask_stream.extend(sample['mask'])
                # Add mask for EOS token (valid = 1)
                if self.eos_id is not None:
                    mask_stream.append(1)

            # Yield complete sequences
            while len(token_stream) >= self.sequence_length + 1:
                # Extract sequence + 1 for input/target
                sequence_tokens = np.array(token_stream[:self.sequence_length + 1])
                seqlen = seqlen_stream
                seqlen[-1] = seqlen[-1] - ( sum(seqlen) - (self.sequence_length + 1))
                assert sum(seqlen) == self.sequence_length + 1
                token_stream = token_stream[self.sequence_length + 1:]
                seqlen_stream = [len(token_stream)]

                output_sample = {
                    'tokens': [sequence_tokens],
                    'seqlen': seqlen
                }


                if has_masks:
                    sequence_masks = np.array(mask_stream[:self.sequence_length + 1])
                    mask_stream = mask_stream[self.sequence_length + 1:]
                    output_sample['mask'] = [sequence_masks]

                yield output_sample

        # Handle remaining tokens
        if not self.drop_last and len(token_stream) >= self.min_sequence_length:
            output_sample = {
                'tokens': [np.array(token_stream)],
                'seqlen': seqlen_stream
            }
            if has_masks and mask_stream:
                output_sample['mask'] = [np.array(mask_stream)]
            yield output_sample