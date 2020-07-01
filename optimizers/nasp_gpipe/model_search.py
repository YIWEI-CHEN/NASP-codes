import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from optimizers.nasp_gpipe.model_search_darts import Network, MixedOp, ChoiceBlock, Cell
from optimizers.darts.operations import *
from optimizers.darts.genotypes import PRIMITIVES


class NASPNetwork(Network):

    def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
        super(NASPNetwork, self).__init__(C, num_classes, layers, criterion, output_weights,
                                          search_space, steps=steps)
        self.saved_params = []
        for w in self._arch_parameters:
            temp = w.data.clone()
            self.saved_params.append(temp)

    def _initialize_alphas(self):
        # Initializes the weights for the mixed ops.
        self._arch_parameters = []
        for _ in range(self._layers):
            num_ops = len(PRIMITIVES)
            alphas_mixed_op = Variable(torch.ones(self._steps, num_ops)/2 +
                                            1e-3*torch.randn(self._steps, num_ops), requires_grad=True)

            # For the alphas on the output node initialize a weighting vector for all choice blocks and the input edge.
            alphas_output = Variable(torch.ones(1, self._steps + 1)/2 +
                                          1e-3*torch.randn(1, self._steps + 1), requires_grad=True)

            if type(self.search_space) == SearchSpace1:
                begin = 3
            else:
                begin = 2
            # Initialize the weights for the inputs to each choice block.
            alphas_inputs = [Variable(torch.ones(1, n_inputs)/2 + 1e-3*torch.randn(1, n_inputs),
                                           requires_grad=True) for n_inputs in range(begin, self._steps + 1)]
            # Total architecture parameters
            _arch_parameters = [
                alphas_mixed_op,
                alphas_output,
                *alphas_inputs
            ]
            self.len_arch_param = len(_arch_parameters)
            self._arch_parameters.extend(_arch_parameters)

    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
            self.saved_params[index].copy_(value.data)
    
    def clip(self):
        clip_scale = []
        m = nn.Hardtanh(0, 1)
        for index in range(len(self._arch_parameters)):
            clip_scale.append(m(Variable(self._arch_parameters[index].data)))
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = clip_scale[index].data

    def binarization(self, e_greedy=0):
        self.save_params()
        # Use binarize only for the mixop, because the rest very quickly gave exploding gradients
        for l in range(self._layers):
            m, n = self._arch_parameters[l*self.len_arch_param].size()
            if np.random.rand() <= e_greedy:
                maxIndexs = np.random.choice(range(n), m)
            else:
                maxIndexs = self._arch_parameters[l*self.len_arch_param].data.cpu().numpy().argmax(axis=1)
            self._arch_parameters[l*self.len_arch_param].data = self.proximal_step(
                self._arch_parameters[l*self.len_arch_param], maxIndexs)
            
    def restore(self):
        for l in range(self._layers):
            device = self._arch_parameters[l*self.len_arch_param].device
            self._arch_parameters[l*self.len_arch_param].data = \
                self.saved_params[l*self.len_arch_param].to(device, non_blocking=True)
    
    def proximal_step(self, var, maxIndexs=None):
        values = torch.zeros_like(var)
        m, n = values.shape
        # alphas = []
        for i in range(m):
            for j in range(n):
                if j == maxIndexs[i]:
                    # alphas.append(values[i][j].copy())
                    values[i][j] = 1
                else:
                    values[i][j] = 0
        return values

