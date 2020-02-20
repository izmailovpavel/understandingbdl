import math
import numpy as np
import torch
from ..utils import extract_parameters, set_weights, train_epoch


class VIFFGModel(torch.nn.Module):
    def __init__(self, base, init_inv_softplus_sigma=-3.0, 
                 prior_log_sigma=3.0, eps=1e-6, with_mu=True, *args, **kwargs):
        super(VIFFGModel, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.rank = sum([param.numel() for param in self.base_model.parameters()])
#         self.base_params = extract_parameters(self.base_model)

        self.prior_log_sigma = prior_log_sigma
        self.eps = eps

        self.with_mu = with_mu
        if with_mu:
            self.mu = torch.nn.Parameter(torch.zeros(self.rank))
        self.inv_softplus_sigma = torch.nn.Parameter(torch.empty(self.rank).fill_(init_inv_softplus_sigma))

    def forward(self, *args, **kwargs):
        device = self.inv_softplus_sigma.device
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma) + self.eps
        if self.with_mu:
            w = self.mu + torch.randn(self.rank, device=device) * sigma
        else:
            w = torch.randn(self.rank, device=device) * sigma

#         set_weights(self.base_params, w, device)
        set_weights(self.base_model, w, device)
        
        return self.base_model(*args, **kwargs)

    def sample(self):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma.detach().cpu()) + self.eps
        w = torch.randn(self.rank) * sigma
        if self.with_mu:
            w += self.mu.detach().cpu()
        return w

    def compute_kl(self):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma) + self.eps

        kl = torch.sum(self.prior_log_sigma - torch.log(sigma) +
                       0.5 * (sigma ** 2) / (math.exp(self.prior_log_sigma * 2)))
        if self.with_mu:
            kl += 0.5 * torch.sum(self.mu ** 2) / math.exp(self.prior_log_sigma * 2)
        return kl

    def compute_entropy(self):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma) + self.eps
        return torch.sum(torch.log(sigma))
