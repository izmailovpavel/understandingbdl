import math
import torch
#from ..utils import train_epoch
#from ..utils import set_weights_wgrad as set_weights
from ..utils import extract_parameters, train_epoch
from ..utils import set_weights_old as set_weights


class VIModel(torch.nn.Module):
    def __init__(self, base, subspace, init_inv_softplus_sigma=-3.0, 
                 prior_log_sigma=3.0, eps=1e-6, with_mu=True, *args, **kwargs):
        super(VIModel, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.base_params = extract_parameters(self.base_model)

        self.subspace = subspace
        self.rank = self.subspace.rank

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
            z = self.mu + torch.randn(self.rank, device=device) * sigma
        else:
            z = torch.randn(self.rank, device=device) * sigma
        w = self.subspace(z)

        set_weights(self.base_params, w, device)
        #set_weights(self.base_model, w, device)

        return self.base_model(*args, **kwargs)

    def sample(self, scale=1.):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma.detach()) + self.eps
        z = torch.randn(self.rank) * sigma * scale
        if self.with_mu:
            z += self.mu.detach()
        w = self.subspace(z)
        return w

    def sample_z(self):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma.detach().cpu()) + self.eps
        z = torch.randn(self.rank) * sigma
        if self.with_mu:
            z += self.mu.detach().cpu()
        return z

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
        

class ELBO(object):
    def __init__(self, criterion, num_samples, temperature=1.):
        self.criterion = criterion
        self.num_samples = num_samples
        self.temperature = temperature
        #print("In ELBO, temperature:", temperature)
        #print("In ELBO, num_samples:", num_samples)

    def __call__(self, model, input, target):

        nll, output, _ = self.criterion(model, input, target)
        kl = model.compute_kl() / self.num_samples
        kl *= self.temperature
        loss = nll + kl
        #loss = nll

        return loss, output, {'nll': nll.item(), 'kl': kl.item()}


class BenchmarkVIModel(VIModel):
    # same as a VI model, except with a fit method
    # for ease of benchmarking
    def __init__(self, loader, criterion, epochs, base, subspace, init_inv_softplus_sigma=-3.0, 
                 prior_log_sigma=3.0, eps=1e-6, with_mu=True, lr=0.01, num_samples=45000, temperature=1.0, use_cuda=True, *args, **kwargs):
        super(BenchmarkVIModel, self).__init__(base, subspace, init_inv_softplus_sigma=-3.0, 
                 prior_log_sigma=prior_log_sigma, eps=eps, with_mu=with_mu, *args, **kwargs)
        
        self.use_cuda = use_cuda
        self.loader = loader
        self.criterion = criterion
        #print("Num Samples ELBO:", num_samples)
        self.optimizer = torch.optim.Adam([param for param in self.parameters()], lr=lr)
        self.elbo = ELBO(self.criterion, num_samples, temperature=temperature)
        self.epochs = epochs

    def fit(self, *args, **kwargs):
        for epoch in range(self.epochs):
            train_res = train_epoch(self.loader, self, self.elbo, self.optimizer, regression=True, cuda = self.use_cuda)
            values = ['%d/%d' % (epoch + 1, self.epochs), train_res['accuracy'], train_res['loss'],
                    train_res['stats']['kl'], train_res['stats']['nll']]
            #print(values)

            #with torch.no_grad():
            #     print("sigma:", torch.nn.functional.softplus(self.inv_softplus_sigma.cpu()))
            #     if self.with_mu:
                    # print("mu:", self.mu.cpu())
