from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE, PARAMS
from genotypes import Genotype
import pdb
import torch.cuda.nvtx as nvtx
from torchgpipe.skip import Namespace, pop, skippable, stash

import update_type

class MixedOp(nn.Module):

  def __init__(self, C, stride, reduction):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    if reduction:
      primitives = PRIMITIVES_REDUCE
    else:
      primitives = PRIMITIVES_NORMAL
    for primitive in primitives:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, updateType):
    if updateType == "weights":
      result = [w * op(x) if w.data else w for w, op in zip(weights, self._ops)]
    else:
      result = [w * op(x) for w, op in zip(weights, self._ops)]
    return sum(result)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, weights):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.weights = weights

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, reduction)
        self._ops.append(op)

  def forward(self, s0_s1):
    s0, s1 = s0_s1
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    self.weights.data = self.weights.to(s0.device, non_blocking=True)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, self.weights[offset+j], update_type.value) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


@skippable(stash=['s0'])
class Prev(nn.Module):
  def forward(self, tensor):
    yield stash('s0', tensor)
    return tensor


@skippable(pop=['s0'])
class PrevPrev(nn.Module):
  def forward(self, s1):
    s0 = yield pop('s0')
    return s0, s1


class FirstPrevPrev(nn.Module):
  def forward(self, tensor):
    return tensor, tensor


class LastPrev(nn.Module):
  def forward(self, tensor):
    return tensor


class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).
    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
      super(Flatten, self).__init__()
      self.start_dim = start_dim
      self.end_dim = end_dim

    def forward(self, input):
      return input.flatten(self.start_dim, self.end_dim)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, gpus, micro_batch_ratio, greedy=0, l2=0, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    # self._criterion = criterion
    self._greedy = greedy
    self._l2 = l2
    self._steps = steps
    self._multiplier = multiplier
    self._init_devices(gpus)
    self._micro_batch_ratio = micro_batch_ratio
    self._initialize_alphas()

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.Sequential()
    all_ns = [Namespace() for _ in range(self._layers)]
    reduction_prev = False
    for i in range(layers):
      weights = self._arch_parameters[i]
      current_ns = all_ns[i]
      prev_ns = all_ns[i-1]
      if self.is_reduce(i):
        C_curr *= 2
        reduction = True
        name = 'reduce_{}'.format(i)
      else:
        reduction = False
        name = 'normal_{}'.format(i)
      if i == self._layers - 1:
        self.cells.add_module('{}_s1'.format(name), LastPrev())
      else:
        self.cells.add_module('{}_s1'.format(name), Prev().isolate(current_ns))
      if i == 0:
        self.cells.add_module('{}_s0'.format(name), FirstPrevPrev())
      else:
        self.cells.add_module('{}_s0'.format(name), PrevPrev().isolate(prev_ns))

      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, weights)
      self.cells.add_module(name, cell)
      reduction_prev = reduction
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.flat = Flatten()
    self.classifier = nn.Linear(C_prev, num_classes)


  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, updateType="weights"):
    split_size = int(input.size(0) * self._micro_batch_ratio)
    splits = iter(input.split(split_size, dim=0))
    s_next = next(splits)
    nvtx.range_push('stem:0')
    s0 = s1 = self.stem(s_next)
    nvtx.range_pop()
    s0_prev, s1_prev = self._block_forward(s0, s1, updateType, 0)

    res = []
    for s_next in splits:
      for device_idx in range(1, len(self.devices)):
        s0_prev, s1_prev = self._block_forward(s0_prev, s1_prev, updateType, device_idx)
      out = self.global_pooling(s1_prev)
      res.append(self.classifier(out.view(out.size(0), -1)))

      # it should run concurrently
      nvtx.range_push('stem:1')
      s0 = s1 = self.stem(s_next)
      nvtx.range_pop()
      s0_prev, s1_prev = self._block_forward(s0, s1, updateType, 0)

    for device_idx in range(1, len(self.devices)):
      s0_prev, s1_prev = self._block_forward(s0_prev, s1_prev, updateType, device_idx)
    out = self.global_pooling(s1_prev)
    res.append(self.classifier(out.view(out.size(0), -1)))

    logits = torch.cat(res)
    return logits

  def _block_forward(self, s0, s1, updateType, device_idx):
    for i in range(device_idx * self.layers_per_dev, (device_idx + 1) * self.layers_per_dev):
      cell = self.cells[i]
      weights = self._arch_parameters[i]
      nvtx.range_push('cell_{}'.format(i))
      s0, s1 = s1, cell(s0, s1, weights, updateType)
      nvtx.range_pop()
    if device_idx != len(self.devices) - 1:
      device = self.devices[device_idx + 1]
      s0 = s0.to(device, non_blocking=True)
      s1 = s1.to(device, non_blocking=True)
    return s0, s1

  def _loss(self, input, target, updateType):
    logits = self(input, updateType)
    return self._criterion(logits, target)
  
  def _l2_loss(self):
    normal_burden = []
    params = 0
    for key in PRIMITIVES_NORMAL:
      params += PARAMS[key]
    for key in PRIMITIVES_NORMAL:
      normal_burden.append(PARAMS[key]/params)
    normal_burden = torch.autograd.Variable(torch.Tensor(normal_burden).cuda(), requires_grad=False)
    return (self.alphas_normal*self.alphas_normal*normal_burden).sum()*self._l2

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops_normal = len(PRIMITIVES_NORMAL)
    num_ops_reduce = len(PRIMITIVES_REDUCE)
    self._arch_parameters = []
    for i in range(self._layers):
      # device = self.devices[i // self.layers_per_dev]
      if self.is_reduce(i):
        alpha = torch.full((k, num_ops_reduce), 0.5, requires_grad=True)
      else:
        alpha = torch.full((k, num_ops_normal), 0.5, requires_grad=True)
      self._arch_parameters.append(alpha)

  def save_params(self):
    self.saved_params = []
    for index, arch in enumerate(self._arch_parameters):
      self.saved_params.append(arch.clone().detach())

  def clip(self):
    m = nn.Hardtanh(0, 1)
    for index in range(len(self._arch_parameters)):
      self._arch_parameters[index].data = m(self._arch_parameters[index].data)

  def binarization(self, e_greedy=0):
    self.save_params()
    for index in range(len(self._arch_parameters)):
      m,n = self._arch_parameters[index].size()
      if np.random.rand() <= e_greedy:
        maxIndexs = np.random.choice(range(n), m)
      else:
        maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)
      self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index], maxIndexs)

  def restore(self):
    for index in range(len(self._arch_parameters)):
      device = self._arch_parameters[index].device
      self._arch_parameters[index].data = self.saved_params[index].to(device, non_blocking=True)

  def proximal_step(self, var, max_indexes):
    m, n = var.shape
    values = torch.zeros_like(var)
    alphas = var[torch.arange(m), max_indexes].data.cpu().numpy()

    step = 2
    cur = 0
    active_rows = []
    active_cols = []
    while cur < m:
      cur_alphas = alphas[cur:cur+step]
      cur_max_index = max_indexes[cur:cur+step]
      sorted_alphas = sorted(list(zip(range(step), cur_alphas, cur_max_index)), key=lambda x:x[1], reverse=True)
      active_rows.extend([v[0] + cur for v in sorted_alphas[:2]])
      active_cols.extend([v[2] for v in sorted_alphas[:2]])
      cur = cur + step
      step += 1
    values[active_rows, active_cols] = 1.0
    return values

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, primitives):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          gene.append((primitives[k_best], j))
        start = end
        n += 1
      return gene

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    GenotypeN = namedtuple('Genotype', 'normal normal_concat')
    GenotypeR = namedtuple('Genotype', 'reduce reduce_concat')
    genotypes = []
    for i, alpha in enumerate(self._arch_parameters):
      if self.is_reduce(i):
        gene = _parse(F.softmax(alpha, dim=-1).data.cpu().numpy(), PRIMITIVES_REDUCE)
        genotype = GenotypeR(reduce=gene, reduce_concat=concat)
      else:
        gene = _parse(F.softmax(alpha, dim=-1).data.cpu().numpy(), PRIMITIVES_NORMAL)
        genotype = GenotypeN(normal=gene, normal_concat=concat)
      genotypes.append(genotype)
    return genotypes

  def _init_devices(self, gpus):
    num_gpus = len(gpus.split(','))
    self.devices = []
    for i in range(num_gpus):
      self.devices.append(torch.device('cuda:{}'.format(i)))

  def is_reduce(self, index):
    return index in [self._layers // 3, 2 * self._layers // 3]
