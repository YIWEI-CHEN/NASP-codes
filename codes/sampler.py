import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import math


class DistributedSubsetSampler(DistributedSampler):
  def __iter__(self):
    # deterministically shuffle based on epoch
    g = torch.Generator()
    g.manual_seed(self.epoch)
    num_dataset = len(self.dataset)
    if self.split is not None:
      num_dataset = self.split
    indices = torch.randperm(num_dataset, generator=g).tolist()

    # add extra samples to make it evenly divisible
    indices += indices[:(self.total_size - len(indices))]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    return iter(indices)

  def set_split(self, split):
    self.num_samples = int(math.ceil(split * 1.0 / self.num_replicas))
    self.total_size = self.num_samples * self.num_replicas
    self.split = split


class SubsetSampler(Sampler):
  r"""Samples elements from a given list of indices, without replacement.

  Arguments:
      indices (sequence): a sequence of indices
  """

  def __init__(self, indices):
    self.indices = indices

  def __iter__(self):
    return (self.indices[i] for i in range(len(self.indices)))

  def __len__(self):
    return len(self.indices)