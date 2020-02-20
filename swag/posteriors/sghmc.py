import math
import numpy as np
import torch
from torch.autograd import Variable

from ..utils import set_weights

class SGHMCModel(torch.nn.Module):
    def __init__(self, base, subspace, mean_nll,
                 num_samples, prior_log_sigma=3.0,
                 alpha=0.01, eta=5e-3, *args, **kwargs):
        super(SGHMCModel, self).__init__()

        self.base_model = base(*args, **kwargs)

        self.mean_nll = mean_nll
        self.num_samples = num_samples

        self.rank = subspace.cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma

        self.subspace = subspace

        self.alpha, self.eta = alpha, eta

        self.t = torch.nn.Parameter(torch.zeros(self.rank))

        self.optimizer = torch.optim.SGD([self.t], lr=1., momentum=(1 - self.alpha))

    def forward(self, *args, **kwargs):
        w = self.subspace(self.t)
        set_weights(self.base_model, w, self.t.device)
        return self.base_model(*args, **kwargs)

    def step(self, inpt, target):
        nll, output, _ = self.mean_nll(self, inpt, target)
        loss = self.eta * nll
        loss -= self.eta * self.log_prior() / self.num_samples
        loss += self._noise()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _noise(self):
        std = np.sqrt(2 * self.alpha * self.eta)
        n = Variable(torch.normal(0, std=std*torch.ones_like(self.t)))
        return torch.sum(n * self.t) 

    def log_prior(self):
        
        sigma = math.exp(self.prior_log_sigma)

        return -torch.sum(self.t**2) / (2 * sigma**2)
