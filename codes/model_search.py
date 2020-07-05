import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE, PARAMS
from genotypes import Genotype
import pdb

from torch.nn import Parameter


class MixedOp(nn.Module):

  def __init__(self, C, stride, reduction, name):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    if reduction:
      primitives = PRIMITIVES_REDUCE
    else:
      primitives = PRIMITIVES_NORMAL
    self.name = name
    self.stride = stride
    for primitive in primitives:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, updateType, active_id):
    if updateType == "weights":
      result = [w * op(x) if w.data.cpu().numpy() else w for w, op in zip(weights, self._ops)]
      return sum(result)
    else:
      def run_function(candidate_ops, active_id, stride, name):
        def forward(_x):
          # print('Forward: {}, active: {}'.format(name, active_id))
          if active_id is None:
            return Zero(stride)(_x)
          return candidate_ops[active_id](_x)
        return forward

      def backward_function(candidate_ops, active_id, binary_gates, name):
        def backward(_x, _output, grad_output):
          binary_grads = torch.zeros_like(binary_gates.data)
          # print('Backward: {}, active: {}'.format(name, active_id))
          with torch.no_grad():
            for k in range(len(candidate_ops)):
              if k != active_id:
                out_k = candidate_ops[k](_x.data)
              else:
                out_k = _output.data
              grad_k = torch.sum(out_k * grad_output)
              binary_grads[k] = grad_k
          return binary_grads

        return backward

      output = ArchGradientFunction.apply(
        x, weights, run_function(self._ops, active_id, self.stride, self.name),
        backward_function(self._ops, active_id, weights, self.name)
      )
      return output


def detach_variable(inputs):
  if isinstance(inputs, tuple):
    return tuple([detach_variable(x) for x in inputs])
  else:
    x = inputs.detach()
    x.requires_grad = inputs.requires_grad
    return x


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
      ctx.run_func = run_func
      ctx.backward_func = backward_func

      detached_x = detach_variable(x)
      with torch.enable_grad():
        output = run_func(detached_x)
      ctx.save_for_backward(detached_x, output)
      return output.data

    @staticmethod
    def backward(ctx, grad_output):
      detached_x, output = ctx.saved_tensors

      grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
      # compute gradients w.r.t. binary_gates
      binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

      return grad_x[0], binary_grads, None, None


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, name):
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
        op = MixedOp(C, stride, reduction, '{}_{}'.format(name, i))
        self._ops.append(op)

  def forward(self, s0, s1, weights, updateType, active_index):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j], updateType, active_index[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, greedy=0, l2=0, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._greedy = greedy
    self._l2 = l2
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
        name = 'R{}'.format(i)
      else:
        reduction = False
        name = 'N{}'.format(i)
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, name)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self.saved_params = []

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, updateType="weights"):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = self.alphas_reduce_binary
        active_index = self._active_index[1]
      else:
        weights = self.alphas_normal_binary
        active_index = self._active_index[0]
      s0, s1 = s1, cell(s0, s1, weights, updateType, active_index)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target, updateType):
    logits = self(input, updateType)
    return self._criterion(logits, target) + self._l2_loss()
  
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
    self.alphas_normal = Parameter(torch.ones(k, num_ops_normal)/2, requires_grad=False)
    self.alphas_reduce = Parameter(torch.ones(k, num_ops_reduce)/2, requires_grad=False)
    self.alphas_normal_binary = Parameter(torch.Tensor(k, num_ops_normal), requires_grad=True)
    self.alphas_reduce_binary = Parameter(torch.Tensor(k, num_ops_reduce), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]
    self._arch_parameters_binary = [
      self.alphas_normal_binary,
      self.alphas_reduce_binary,
    ]

    self._active_index = [
      [None for _ in range(k)],
      [None for _ in range(k)]
    ]

  def clip(self):
    clip_scale = []
    m = nn.Hardtanh(0, 1)
    for index in range(len(self._arch_parameters)):
      clip_scale.append(m(Variable(self._arch_parameters[index].data)))
    for index in range(len(self._arch_parameters)):
      self._arch_parameters[index].data = clip_scale[index].data

  def binarization(self, e_greedy=0):
    for index in range(len(self._arch_parameters)):
      m,n = self._arch_parameters[index].size()
      if np.random.rand() <= e_greedy:
        maxIndexs = np.random.choice(range(n), m)
      else:
        maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)
      self.proximal_step(self._arch_parameters[index], maxIndexs, index)

  def restore(self):
    m = len(self._active_index[0])
    for index in range(len(self._arch_parameters)):
      self._arch_parameters_binary[index].data.zero_()
      self._arch_parameters_binary[index].grad = None
      self._active_index[index] = [None for _ in range(m)]

  def proximal_step(self, var, max_indexes=None, index=0):
    m, n = var.shape
    self._arch_parameters_binary[index].data.zero_()
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
    for row, col in zip(active_rows, active_cols):
      self._active_index[index][row] = col
    self._arch_parameters_binary[index].data[active_rows, active_cols] = 1.0

  def arch_parameters(self):
    return self._arch_parameters

  def arch_parameters_binary(self):
    return self._arch_parameters_binary

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
