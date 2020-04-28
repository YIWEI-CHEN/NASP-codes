import re
import time

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def get_gpu_time(prof):
  m = re.search("(?<=CUDA time total:) ([0-9]{1,4}\.[0-9]{1,4})(ms|s)", str(prof))
  unit = m.group(2)
  if unit == 'ms':
    return float(m.group(1)) / 1000.0
  return float(m.group(1))

class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid, updateType="alphas")
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid, updateType):
    self.model.binarization()

    with torch.autograd.profiler.profile(use_cuda=True) as forward_prof:
      # begin = time.time()
      loss = self.model._loss(input_valid, target_valid, updateType)
      # end = time.time()
      # self.alpha_forward += end - begin

    # print(forward_prof.key_averages().table(sort_by="cuda_time_total"))]
    self.alpha_forward_cpu += forward_prof.self_cpu_time_total / 1e6
    self.alpha_forward += get_gpu_time(forward_prof)

    with torch.autograd.profiler.profile(use_cuda=True) as backward_prof:
      # begin = time.time()
      grad = torch.autograd.grad(loss, self.model.arch_parameters())
      for i, arch in enumerate(self.model.arch_parameters()):
        arch.grad = grad[i]
      # end = time.time()
      # self.alpha_backward += end - begin
    self.alpha_backward_cpu += backward_prof.self_cpu_time_total / 1e6
    self.alpha_backward += get_gpu_time(backward_prof)
    # print(backward_prof.key_averages().table(sort_by="cuda_time_total"))

    self.model.restore()
