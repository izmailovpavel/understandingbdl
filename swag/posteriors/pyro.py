import math
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS, MCMC
from pyro.nn import AutoRegressiveNN
from torch.autograd import Variable

from ..utils import extract_parameters, set_weights

class PyroModel(torch.nn.Module):

    def __init__(self, 
                 base, 
                 subspace,
                 prior_log_sigma, 
                 likelihood_given_outputs,
                 batch_size = 100,
                 *args, **kwargs):

        super(PyroModel, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.base_params = extract_parameters(self.base_model)

        #self.rank = cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma
        
        self.likelihood = likelihood_given_outputs
        self.batch_size = batch_size
        self.subspace = subspace
        self.rank = self.subspace.cov_factor.size(0)
    
    def model(self, x, y):

        self.t = pyro.sample("t", dist.Normal(torch.zeros(self.rank), 
                                  torch.ones(self.rank) * np.exp(self.prior_log_sigma)).to_event(1))
        self.t = self.t.to(x.device)

        bs = self.batch_size
        num_batches = x.shape[0] // bs
        if x.shape[0] % bs: num_batches += 1

        for i in pyro.plate("batches", num_batches): 
            
            x_ = x[i * bs: (i+1)*bs] 
            y_ = y[i * bs: (i+1)*bs] 

            with pyro.plate("data"+str(i), x_.shape[0]):
                w = self.subspace(self.t)
                set_weights(self.base_params, w, self.t.device)
                z = self.base_model(x_)
                pyro.sample("y"+str(i), self.likelihood(z).to_event(1), obs=y_)

    def model_subsample(self, x, y):
        subsample_size = self.batch_size
        self.t = pyro.sample("t", dist.Normal(torch.zeros(self.rank), 
                                  torch.ones(self.rank) * np.exp(self.prior_log_sigma)).to_event(1))
        self.t = self.t.to(x.device)
            
        with pyro.plate("data", x.shape[0], subsample_size=subsample_size) as ind:
            #w = self.mean.to(self.t.device) + self.cov_factor.to(self.t.device).t() @ self.t
            w = self.subspace(self.t)
            set_weights(self.base_params, w, self.t.device)
            z = self.base_model(x[ind])
            pyro.sample("y", self.likelihood(z).to_event(1), obs=y[ind])

    def forward(self, *args, **kwargs):
        w = self.subspace(self.t)
        set_weights(self.base_params, w, self.t.device)
        return self.base_model(*args, **kwargs)

class BenchmarkPyroModel(PyroModel):
    def __init__(self, base, subspace, prior_log_sigma, likelihood_given_outputs, batch_size = 100,
                kernel=NUTS, num_samples=30, kernel_kwargs={}, 
                *args, **kwargs):
        super(BenchmarkPyroModel, self).__init__(base, subspace, prior_log_sigma, likelihood_given_outputs, 
            batch_size=batch_size, *args, **kwargs)
        self.kernel = kernel(self.model, **kernel_kwargs)
        self.num_samples = num_samples

        #self.loader = loader
        self.all_samples = None
        self.mcmc_run = None

    def fit(self, inputs, targets, *args, **kwargs):
        self.mcmc_run = MCMC(self.kernel, num_samples=self.num_samples, warmup_steps=100).run(inputs, targets)
        self.all_samples = torch.cat(list(self.mcmc_run.marginal(sites="t").support(flatten=True).values()), dim=-1)

    def sample(self, ind=None, scale=1.0):
        if ind is None:
            ind = np.random.randint(self.num_samples)
        self.eval()
        
        self.t.set_(self.all_samples[int(ind), :])

class GaussianGuide(torch.nn.Module):

    def __init__(self, rank, init_inv_softplus_sigma=-3.0, eps=1e-6, with_mu=True):
        super(GaussianGuide, self).__init__()

        self.rank = rank
        self.eps = eps
        self.with_mu = with_mu
        self.init_inv_softplus_sigma = init_inv_softplus_sigma

    def model(self, *args, **kargs):

        if self.with_mu:
            self.mu = pyro.param("mu", torch.zeros(self.rank))
        else: 
            self.mu = torch.zeros(self.rank)
        self.inv_softplus_sigma = pyro.param("inv_softplus_sigma", 
                         torch.ones(self.rank)*(self.init_inv_softplus_sigma))

        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma) + self.eps
        self.t = pyro.sample("t", dist.Normal(self.mu, sigma).to_event(1))
        return self.t


class IAFGuide(torch.nn.Module):

    def __init__(self, rank, n_hid=[100]):
        super(IAFGuide, self).__init__()

        self.rank = rank
        self.n_hid = n_hid

    @property
    def sigma(self):
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma)
        return sigma

    def model(self, *args, **kargs):

        self.inv_softplus_sigma = pyro.param("inv_softplus_sigma", torch.ones(self.rank))
        sigma = self.sigma#torch.nn.functional.softplus(self.inv_softplus_sigma)

        #base_dist = dist.Normal(torch.zeros(self.rank), torch.ones(self.rank))
        # Pavel: introducing `sigma` in the IAF distribution makes training more
        # stable in tems of the scale of the distribution we are trying to learn
        base_dist = dist.Normal(torch.zeros(self.rank), sigma)
        ann = AutoRegressiveNN(self.rank, self.n_hid, skip_connections=True)
        iaf = dist.InverseAutoregressiveFlow(ann)
        iaf_module = pyro.module("my_iaf", iaf)
        iaf_dist = dist.TransformedDistribution(base_dist, [iaf])
        self.t = pyro.sample("t", iaf_dist.to_event(1))
        return self.t


class TemperedCategorical(dist.Categorical):

    def __init__(self, temperature=1., *args, **kwargs):
        super(TemperedCategorical, self).__init__(*args, **kwargs)
        self.temperature = temperature

    def log_prob(self, value):
        ans = super(TemperedCategorical, self).log_prob(value)
        return ans / self.temperature
        
    def expand(self, batch_shape):
        # Blindly copied from pyro
        batch_shape = torch.Size(batch_shape)
        validate_args = self.__dict__.get('validate_args')
        if 'probs' in self.__dict__:
            probs = self.probs.expand(batch_shape + self.probs.shape[-1:])
            return TemperedCategorical(temperature=self.temperature, probs=probs, validate_args=validate_args)
        else:
            logits = self.logits.expand(batch_shape + self.logits.shape[-1:])
            return TemperedCategorical(temperature=self.temperature, logits=logits, validate_args=validate_args)
