import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from optimizers.nasp_gpipe import update_type


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class ArchitectNASP(object):

    def __init__(self, model, criterion, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self._criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), 
                                          weight_decay=args.arch_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        self.model.binarization()
        update_type.set_alphas()
        logits = self.model(input_valid)
        loss = self._criterion(logits, target_valid)
        loss.backward()
        self.model.restore()
