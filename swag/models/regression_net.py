import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms


__all__ = [
    'RegNet',
    'ToyRegNet',
]


class MDropout(torch.nn.Module):
    def __init__(self, dim, p, inplace=False):
        super(MDropout, self).__init__()
        self.dim = dim
        self.p = p
        self.inplace = inplace
        self.register_buffer('mask', torch.ones(dim, dtype=torch.long))
    
    def forward(self, input):
        if self.training:
            return torch.nn.functional.dropout(input, self.p, self.training, self.inplace)
        else:
            return input * self.mask.float().view(1, -1) * 1.0 / (1.0 - self.p)
        
    def sample(self):        
        self.mask.bernoulli_(1.0 - self.p) 


def sample_masks(module):    
    if isinstance(module, MDropout):        
        module.sample()

class SplitDim(nn.Module):
    def __init__(self, nonlin_col=1, nonlin_type=torch.nn.functional.softplus, correction = True):
        super(SplitDim, self).__init__()
        self.nonlinearity = nonlin_type
        self.col = nonlin_col

        if correction:
            self.var = torch.nn.Parameter(torch.zeros(1))
        else:
            #equivalent to about 3e-7 when using softplus
            self.register_buffer('var', torch.ones(1, requires_grad = False)*-15.)

        self.correction = correction

    def forward(self, input):
        transformed_output = self.nonlinearity(input[:,self.col])
        
        transformed_output = (transformed_output + self.nonlinearity(self.var))
        stack_list = [input[:,:self.col], transformed_output.view(-1,1)]
        if self.col+1 < input.size(1):
            stack_list.append(input[:,(self.col+1):])
        
        #print(self.nonlinearity(self.var).item(), transformed_output.mean().item())
        output = torch.cat(stack_list,1)
        return output


class RegNetBase(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=1, dropout=None, apply_var=True):
        super(RegNetBase, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]        
        for i in range(len(self.dimensions) - 1):
            if dropout is not None and i > 0:
                self.add_module('dropout%d' % i, MDropout(self.dimensions[i], p=dropout))
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())

        if output_dim == 2:
            self.add_module('var_split', SplitDim(correction=apply_var))

    def forward(self, x, output_features=False):
        if not output_features:
            return super().forward(x)
        else:
            print(self._modules.values())
            print(list(self._modules.values())[:-2])
            for module in list(self._modules.values())[:-3]:
                x = module(x)
                print(x.size())
            return x

class RegNetCurve(nn.Sequential):

    def __init__(self, dimensions, fix_points, dropout=None):
        super(RegNetCurve, self).__init__()
        self.dimensions = dimensions        
        for i in range(len(dimensions) - 1):
            if dropout is not None and i > 0:
                self.add_module('dropout%d' % i, MDropout(dimensions[i], p=dropout))
            self.add_module('linear%d' % i, curves.Linear(dimensions[i], dimensions[i + 1], fix_points=fix_points))
            if i < len(dimensions) - 2:
                self.add_module('tanh%d' % i, torch.nn.Tanh())

    def forward(self, x, t):
        for module in self._modules.values():
            if isinstance(module, curves.Linear):
                x = module(x, t)
            else:
                x = module(x)
                
        return x


class RegNet:
    base = RegNetBase
    curve = RegNetCurve
    args = list()
    kwargs = {"dimensions": [1000, 1000, 500, 50, 2]}

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])


class ToyRegNet:
    base = RegNetBase
    curve = RegNetCurve
    args = list()
    kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 2} 

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])


