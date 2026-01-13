
import numpy as np
from bisect import insort_left
from collections import deque
from typing import List, Tuple, Iterator, Optional
from torch.utils.data import IterableDataset
import multiprocessing as mp
import torch



class ContinualSequencer(IterableDataset):

    def __init__(self, dataset: IterableDataset, sequence_length: int,
                 min_sequence_length: int = 0, drop_last: bool = False):

        self.dataset = dataset  # Store the dataset, don't convert to iterator yet
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.drop_last = drop_last

    def _get_sample_from_dataset(self, dataset_iter):
        while True:
            try:
                sample = next(dataset_iter)
                if len(sample['tokens']) > self.min_sequence_length:
                    return sample
            except StopIteration:
                raise StopIteration

    def _split_sample(self, sample: dict, length: int) -> Tuple[dict, dict]:
        if isinstance(sample['tokens'], list):
            len_prev = sum([s.shape[0] for s in sample['tokens'][:-1]])

            last_seq = sample['tokens'][-1]
            if 'mask' in sample:
                last_mask = sample['mask'][-1]
                sample_a_pre = {'tokens': sample['tokens'][:-1], 'mask': sample['mask'][:-1]}
                sample_a_last, sample_b = self._split_sample({'tokens': last_seq, 'mask': last_mask}, length - len_prev)
            else:
                sample_a_pre = {'tokens': sample['tokens'][:-1]}
                sample_a_last, sample_b = self._split_sample({'tokens': last_seq}, length - len_prev)

            sample_a = self._merge_samples(sample_a_pre, sample_a_last)
        else:
            sample_a = {}
            sample_b = {}
            for key, value in sample.items():
                if key == "seqlens":
                    sample_a[key] = [l for l in value if l < length]
                    sample_b[key] = [l - length for l in value if l >= length]
                else:
                    sample_a[key] = value[:length]
                    sample_b[key] = value[length:]

            if len(sample_b['tokens']) < self.min_sequence_length:
                sample_b = None

        return sample_a, sample_b

    def _merge_samples(self, sample_a: dict, sample_b: dict) -> dict:
        sample = {}

        if not isinstance(sample_a['tokens'], list):
            sample_a['tokens'] = [sample_a['tokens']]
        if not isinstance(sample_b['tokens'], list):
            sample_b['tokens'] = [sample_b['tokens']]

        sample['tokens'] = sample_a['tokens'] + sample_b['tokens']

        if 'mask' in sample_a:
            if not isinstance(sample_a['mask'], list):
                sample_a['mask'] = [sample_a['mask']]
            if not isinstance(sample_b['mask'], list):
                sample_b['mask'] = [sample_b['mask']]
            sample['mask'] = sample_a['mask'] + sample_b['mask']

        return sample

    @staticmethod
    def sl(sample):  # sample length
        if isinstance(sample['tokens'], list):
            return sum([s.shape[0] for s in sample['tokens']])
        else:
            return sample['tokens'].shape[0]

    @staticmethod
    def ss(sample):  # sets in sample
        if isinstance(sample['tokens'], list):
            return len(sample['tokens'])
        else:
            return 1

    def _continual_sample_generator(self, dataset_iter):
        """Generator that yields processed samples"""
        sample = None
        left_over = None
        stop_flag = False

        while not stop_flag:
            if sample is not None:
                if self.sl(sample) == self.sequence_length + self.ss(sample):
                    yield sample
                    sample = None

                elif self.sl(sample) > self.sequence_length + self.ss(sample):
                    sample, left_over = self._split_sample(sample, self.sequence_length + self.ss(sample))
                    yield sample
                    sample = None

                elif self.sequence_length - self.min_sequence_length <= self.sl(sample) < self.sequence_length + self.ss(sample):
                    sample, _ = self._split_sample(sample, self.sequence_length-self.min_sequence_length-1)

                elif self.sl(sample) < self.sequence_length:


                    if left_over is not None:
                        if self.sl(sample) + self.sl(left_over) > self.sequence_length + self.ss(sample) + 1:
                            new_sample, left_over = self._split_sample(left_over, self.sequence_length + self.ss(sample) + 1 - self.sl(sample))
                            sample = self._merge_samples(sample, new_sample)
                        else:
                            if self.sl(sample) + self.sl(left_over) + self.min_sequence_length <= self.sequence_length + self.ss(sample) + 1:
                                sample = self._merge_samples(sample, left_over)
                            left_over = None
                    else:
                        try:
                            new_sample = self._get_sample_from_dataset(dataset_iter)
                        except StopIteration:
                            stop_flag = True

                        if not stop_flag:
                            if self.sl(sample) + self.sl(new_sample) > self.sequence_length + self.ss(sample) + 1:
                                new_sample, left_over = self._split_sample(new_sample, self.sequence_length + self.ss(sample) + 1 - self.sl(sample))
                                sample = self._merge_samples(sample, new_sample)
                            else:
                                if self.sl(sample) + self.sl(new_sample) + self.min_sequence_length <= self.sequence_length + self.ss(sample) + 1:
                                    sample = self._merge_samples(sample, new_sample)
                else:
                    raise ValueError("This should not happen")
            else:
                try:
                    sample = self._get_sample_from_dataset(dataset_iter)
                except StopIteration:
                    stop_flag = True

        if not self.drop_last and sample is not None:
            yield sample

    def __iter__(self) -> Iterator[dict]:
        """Create a new iterator for each worker/epoch"""
        # Create a fresh iterator from the dataset for this iteration
        dataset_iter = iter(self.dataset)

        # Return the generator
        for sample in self._continual_sample_generator(dataset_iter):
            # Ensure tokens is always a list
            if not isinstance(sample['tokens'], list):
                sample['tokens'] = [sample['tokens']]
                if 'mask' in sample:
                    sample['mask'] = [sample['mask']]

            if not self.drop_last:
                assert sum([s.shape[0] for s in sample['tokens']]) <= self.sequence_length + self.ss(sample)
            else:
                assert sum([s.shape[0] for s in sample['tokens']]) == self.sequence_length + self.ss(sample)

            yield sample




