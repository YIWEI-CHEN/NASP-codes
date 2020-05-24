import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from operations import *
from genotypes import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE, PARAMS
from genotypes import Genotype

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

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

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

  def __init__(self, C, num_classes, layers, criterion, greedy=0, l2=0, steps=4, multiplier=4, stem_multiplier=3, gpu=None):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._greedy = greedy
    self._l2 = l2
    self._steps = steps
    self._multiplier = multiplier
    self.device = None if gpu is None else torch.device('cuda:{}'.format(gpu))

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if self.is_reduce(i):
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def forward(self, input, updateType="weights"):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = self.alphas_reduce
      else:
        weights = self.alphas_normal
      s0, s1 = s1, cell(s0, s1, weights, updateType)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target, updateType):
    logits = self(input, updateType)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops_normal = len(PRIMITIVES_NORMAL)
    num_ops_reduce = len(PRIMITIVES_REDUCE)
    self.alphas_normal = torch.full((k, num_ops_normal), 0.5, requires_grad=True, device=self.device)
    self.alphas_reduce = torch.full((k, num_ops_reduce), 0.5, requires_grad=True, device=self.device)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

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

  def proximal_step(self, var, max_indexes=None):
    m, n = var.shape
    values = torch.zeros_like(var)
    alphas = var[torch.arange(m), max_indexes].data.cpu().numpy()

    step = 2
    cur = 0
    active_rows = []
    active_cols = []
    while cur < m:
      cur_alphas = alphas[cur:cur + step]
      cur_max_index = max_indexes[cur:cur + step]
      sorted_alphas = sorted(list(zip(range(step), cur_alphas, cur_max_index)), key=lambda x: x[1], reverse=True)
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

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), PRIMITIVES_NORMAL)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), PRIMITIVES_REDUCE)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def is_reduce(self, index):
    return index in [self._layers // 3, 2 * self._layers // 3]
