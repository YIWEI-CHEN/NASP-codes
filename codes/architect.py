import time

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from model_search import Cell, MixedOp
from operations import DilConv, SepConv, ReLUConvBN, FactorizedReduce

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, name, module, backward=False):
        self.name = name
        self.backward = backward
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
      if self.backward:
        # print('b: {}: {}'.format(self.name, module))
        print('b: {}'.format(self.name))
      else:
        print('f: {}'.format(self.name))
        # print('{}: {}, in({}), out({})'.format(self.name, module, input[0].shape, output.shape))
        # self.input = input
        # self.output = output
        # self.module = module
    def close(self):
        self.hook.remove()


def get_all_layers(net, hooks, backward=False, parent_name=''):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList) or isinstance(layer, Cell):
      name = '{}.{}'.format(parent_name, name) if parent_name != '' else name
      get_all_layers(layer, hooks, backward, parent_name=name)
    # elif isinstance(layer, MixedOp):
    #   # or isinstance(layer, DilConv) or isinstance(layer, SepConv) or \
    #   # isinstance(layer, ReLUConvBN) or isinstance(layer, FactorizedReduce):
    #   name = '{}.{}({})'.format(parent_name, layer.__class__.__name__, name) if parent_name != '' else name
    #   get_all_layers(layer, hooks, backward, parent_name=name)
    else:
      if isinstance(layer, DilConv) or isinstance(layer, SepConv) or \
              isinstance(layer, ReLUConvBN) or isinstance(layer, FactorizedReduce):
        name = '{}.{}.{}'.format(parent_name, name, layer.__class__.__name__) if parent_name != '' else name
      else:
        name = '{}.{}'.format(parent_name, name) if parent_name != '' else name
      hook = Hook(name, layer, backward)
      hooks.append(hook)


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
    # register hooks on each layer
    hookF, hookB = [], []
    get_all_layers(self.model, hookF)
    get_all_layers(self.model, hookB, backward=True)
    # hookF = [Hook(layer[1]) for layer in list(self.model._modules.items())]
    # hookB = [Hook(layer[1], backward=True) for layer in list(self.model._modules.items())]

    self.model.binarization()

    begin = time.time()
    loss = self.model._loss(input_valid, target_valid, updateType)
    end = time.time()
    self.alpha_forward += end - begin

    begin = time.time()
    # loss.backward()
    grad = torch.autograd.grad(loss, self.model.arch_parameters())
    for i, arch in enumerate(self.model.arch_parameters()):
        arch.grad = grad[i]

    end = time.time()
    self.alpha_backward += end - begin

    self.model.restore()
