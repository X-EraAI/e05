from typing import Iterator, Sized, NamedTuple
from math import ceil

import torch
from torch.utils.data import Sampler


class AugmentedIndex(NamedTuple):
    index: int
    sample_seed: int
    batch_seed: int


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int,
                 official_augmentation: bool = False):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.resume_batch_offset = 0
        self.official_augmentation = official_augmentation

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.official_augmentation:
            from e05.aug.official import augmentation_seed
            group_size = self.batch_size * self.num_processes
            total = ceil(len(indices) / group_size) * group_size
            if not indices:
                return iter([])
            indices = (indices * ceil(total / len(indices)))[:total]
            indices = [AugmentedIndex(index,
                       augmentation_seed(self.seed, self.epoch, position, 'sample'),
                       augmentation_seed(self.seed, self.epoch, position // group_size, 'batch'))
                       for position, index in enumerate(indices)]
        if self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        if self.official_augmentation:
            group_size = self.batch_size * self.num_processes
            return ceil(len(self.dataset) / group_size) * group_size
        return len(self.dataset)
