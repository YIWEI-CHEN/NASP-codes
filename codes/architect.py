import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda.nvtx as nvtx

from utils import get_elaspe_time
import update_type

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, criterion, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self._criterion = criterion
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    self.optimizer.zero_grad()
    update_type.set_alphas()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    self.model.binarization()

    begin.record()
    # nvtx.range_push('alpha forward')
    logits = self.model(input_valid)
    loss = self._criterion(logits, target_valid)
    # nvtx.range_pop()
    end.record()
    self.alpha_forward += get_elaspe_time(begin, end)

    begin.record()
    # nvtx.range_push('alpha backward')
    # loss.backward()
    grad = torch.autograd.grad(loss, self.model.arch_parameters())
    for i, arch in enumerate(self.model.arch_parameters()):
        arch.grad = grad[i]
    # nvtx.range_pop()
    end.record()
    self.alpha_backward += get_elaspe_time(begin, end)

    self.model.restore()
