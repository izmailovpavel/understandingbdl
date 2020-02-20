import math
import numpy as np
import torch
from torch import nn
from torch import distributions


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior, device=None):
        super().__init__()

        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

        self.to(device)
        self.device = device

    def g(self, z):
        x = z
        for i in reversed(range(len(self.mask))):
            mx = self.mask[i] * x
            tmx = self.t[i](mx)
            smx = self.s[i](mx)
            x = mx + (1 - self.mask[i]) * ((x - tmx) * torch.exp(-smx))
        return x

    def f(self, x):
        z = x
        log_det_J = 0
        for i in range(len(self.mask)):
            mz = self.mask[i] * z
            smz = self.s[i](mz)
            tmz = self.t[i](mz)
            z = mz + (1 - self.mask[i]) * (z * torch.exp(smz) + tmz)
            if x.dim() == 2:
                log_det_J += (smz * (1-self.mask[i])).sum(1)
            else:
                log_det_J += (smz * (1-self.mask[i])).sum(1, 2, 3)
        return z, log_det_J

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, bs=1):
        z = self.prior.sample(torch.Size([bs]))
        x = self.g(z)
        return x



def construct_flow(D, coupling_layers_num=2, inner_dim=128, inner_layers=2, prior=None, device=None):
    def inner_seq(n, inner_dim):
        res = []
        for _ in range(n):
            res.append(nn.Linear(inner_dim, inner_dim))
            res.append(nn.ReLU())
        return res

    class Nets(nn.Module):
        """ net for parametrizing scaling function in coupling layer """
        def __init__(self, D, inner_dim, inner_layers):
            super().__init__()
            self.seq_part = nn.Sequential(nn.Linear(D, inner_dim),
                                     nn.ReLU(),
                                     *inner_seq(inner_layers, inner_dim),
                                     nn.Linear(inner_dim, D),
                                     nn.Tanh())
            self.scale = nn.Parameter(torch.ones(D))
        def forward(self, x):
            x = self.seq_part.forward(x)
            x = self.scale * x
            return x

    # a function that take no arguments and return a pytorch model, dim(X) -> dim(X)
    nets = lambda: Nets(D, inner_dim, inner_layers)
    nett = lambda: nn.Sequential(nn.Linear(D, inner_dim),
                                 nn.ReLU(),
                                 *inner_seq(inner_layers, inner_dim),
                                 nn.Linear(inner_dim, D))
    
    if prior is None:
        prior = distributions.MultivariateNormal(torch.zeros(D).to(device),
                                                 torch.eye(D).to(device))

    d = D // 2
    masks = torch.zeros(coupling_layers_num, D)
    for i in range(masks.size(0)):
        if i % 2:
            masks[i, :d] = 1.
        else:
            masks[i, d:] = 1.
    masks.to(device)
    return RealNVP(nets, nett, masks, prior, device=device)

