import math
import torch
from swag.utils import set_weights


class VINFModel(torch.nn.Module):
    def __init__(self, base, subspace, flow, 
                 prior_log_sigma=1.0, *args, **kwargs):
        super(VINFModel, self).__init__()

        self.base_model = base(*args, **kwargs)

        self.flow = flow

        self.subspace = subspace
        self.rank = self.subspace.rank
        self.prior_log_sigma = prior_log_sigma

    def forward(self, input, t=None, *args, **kwargs):
        if t is None:
            t = self.flow.sample()
        w = self.subspace(t.squeeze())
        set_weights(self.base_model, w, self.flow.device)
        return self.base_model(input, *args, **kwargs)

    def compute_kl_mc(self, t=None):
        if t is None:
            t = self.flow.sample()
        prior_logprob = - torch.norm(t.squeeze())**2 / (2 * math.exp(self.prior_log_sigma * 2))
        return self.flow.log_prob(t) - prior_logprob

    def compute_entropy_mc(self, t=None):
        if t is None:
             t = self.flow.sample()
        return self.flow.log_prob(t)
        


class ELBO_NF(object):
    def __init__(self, criterion, num_samples, temperature=1.):
        self.criterion = criterion
        self.num_samples = num_samples
        self.temperature = temperature

    def __call__(self, model, input, target):
        # likelihood term
        t = model.flow.sample()
        output = model(input, t=t)
        nll, _ = self.criterion(output, target)

        # kl term
        kl = model.compute_kl_mc(t)
        loss = nll + kl * self.temperature / self.num_samples  # -elbo
        return loss, output, {"nll": nll.item(), "kl": kl.item()}



class BenchmarkVINFModel(VINFModel):
    # same as a VINFModel, except with a fit method
    # for ease of benchmarking
    def __init__(self, loader, criterion, optimizer, epochs, base, subspace, flow,
            prior_log_sigma=3.0, lr=0.1, temperature=1., num_samples=45000, *args, **kwargs):
        super(BenchmarkVINFModel, self).__init__(base, subspace, flow, prior_log_sigma=prior_log_sigma)

        self.loader = loader
        self.criterion = criterion
        self.optimizer = torch.optim.Adam([param for param in self.parameters()], lr=lr)
        self.elbo = ELBO_NF(self.criterion, num_samples, temperature)
    
    def fit(self, *args, **kwargs):
        for epoch in range(self.epochs):
            train_res = train_epoch(self.loader, self, self.elbo, self.optimizer)
            values = ['%d/%d' % (epoch + 1, self.epochs), train_res['accuracy'], train_res['loss'],
                    train_res['stats']['kl'], train_res['stats']['nll']]
            print(values)

