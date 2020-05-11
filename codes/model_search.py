from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE, PARAMS
from genotypes import Genotype
import pdb

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

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, device):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.device = device

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

  def forward(self, s0, s1, weights, updateType):
    # weights = weights.to(self.device)

    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j], updateType) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, gpus, micro_batch_ratio, greedy=0, l2=0, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._greedy = greedy
    self._l2 = l2
    self._steps = steps
    self._multiplier = multiplier
    self._init_devices(gpus)
    self._micro_batch_ratio = micro_batch_ratio

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.stem.to(self.devices[0])
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    self.layers_per_dev = layers // len(self.devices)
    for i in range(layers):
      device = self.devices[i // self.layers_per_dev]
      if self._is_reduce(i):
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, device)
      cell.to(device)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1).to(self.devices[-1])
    self.classifier = nn.Linear(C_prev, num_classes).to(self.devices[-1])

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, updateType="weights"):
    split_size = int(input.size(0) * self._micro_batch_ratio)
    splits = iter(input.split(split_size, dim=0))
    s_next = next(splits)
    s0 = s1 = self.stem(s_next)
    s0_prev, s1_prev = self._block_forward(s0, s1, updateType, 0)

    res = []
    for s_next in splits:
      for device_idx in range(1, len(self.devices)):
        s0_prev, s1_prev = self._block_forward(s0_prev, s1_prev, updateType, device_idx)
      out = self.global_pooling(s1_prev)
      res.append(self.classifier(out.view(out.size(0), -1)))

      # it should run concurrently
      s0 = s1 = self.stem(s_next)
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
      s0, s1 = s1, cell(s0, s1, weights, updateType)
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
      device = self.devices[i // self.layers_per_dev]
      if self._is_reduce(i):
        alpha = torch.full((k, num_ops_reduce), 0.5, requires_grad=True, device=device)
      else:
        alpha = torch.full((k, num_ops_normal), 0.5, requires_grad=True, device=device)
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
      self._arch_parameters[index].data = self.saved_params[index]

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
      if self._is_reduce(i):
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

  def _is_reduce(self, index):
    return index in [self._layers // 3, 2 * self._layers // 3]
