"""
    inferences class w/in the subspace
    currently only fitting the Gaussian associated is implemented
"""

import abc
import torch
import numpy as np

from torch.distributions import LowRankMultivariateNormal
from .elliptical_slice import elliptical_slice
from swag.utils import unflatten_like, flatten, train_epoch
from .proj_model import ProjectedModel
from .vi_model import VIModel, ELBO

class Inference(torch.nn.Module, metaclass=abc.ABCMeta):

    subclasses = {}

    @classmethod
    def register_subclass(cls, inference_type):
        def decorator(subclass):
            cls.subclasses[inference_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, inference_type, **kwargs):
        if inference_type not in cls.subclasses:
            raise ValueError('Bad inference type {}'.format(inference_type))
        return cls.subclasses[inference_type](**kwargs)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super(Inference, self).__init__()

    @abc.abstractmethod
    def fit(self, mean, variance, cov_factor, *args, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass


@Inference.register_subclass('low_rank_gaussian')
class LRGaussian(Inference):

    def __init__(self, base, base_args, base_kwargs, var_clamp=1e-6):
        super(LRGaussian, self).__init__()
        self.var_clamp = var_clamp
        self.dist = None

    def fit(self, mean, variance, cov_factor):
        # ensure variance >= var_clamp
        variance = torch.clamp(variance, self.var_clamp)

        # form a low rank (+ diagonal Gaussian) distribution when fitting
        self.dist = LowRankMultivariateNormal(loc=mean, cov_diag=variance,
                                              cov_factor=cov_factor.t())
    
    def sample(self, scale=0.5, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # x = \mu + L'z
        unscaled_sample = self.dist.rsample()

        # x' = \sqrt(scale) * (x - \mu) + \mu
        scaled_sample = (scale ** 0.5) * (unscaled_sample - self.dist.loc) + self.dist.loc

        return scaled_sample

    def log_prob(self, sample):
        return self.dist.log_prob(sample)


@Inference.register_subclass('projected_sgd')
class ProjSGD(Inference):
    def __init__(self, model, loader, criterion, epochs = 10, **kwargs):
        super(ProjSGD, self).__init__()
        self.kwargs = kwargs
        self.optimizer = None

        self.epochs = epochs

        self.mean, self.var, self.subspace = None, None, None
        self.optimizer = None
        self.proj_params = None
        self.loader, self.criterion = loader, criterion

        self.model = model
    
    def fit(self, mean, variance, subspace, use_cuda = True, **kwargs):
        
        if use_cuda and torch.cuda.is_available():
            self.mean = mean.cuda()
            self.subspace = subspace.cuda()
        else:
            self.mean = mean
            self.subspace = subspace
        
        if self.proj_params is None:
            proj_params = torch.zeros(self.subspace.size(0), 1, dtype = self.subspace.dtype, device = self.subspace.device, requires_grad = True)
            print(proj_params.device)
            self.proj_model = ProjectedModel(model=self.model, mean=self.mean.unsqueeze(1),  projection=self.subspace, proj_params=proj_params)

            # define optimizer
            self.optimizer = torch.optim.SGD([proj_params], **self.kwargs)
        else:
            proj_params = self.proj_params.clone()

        # now train projected parameters
        loss_vec = []
        for _ in range(self.epochs):
            loss = train_epoch(loader=self.loader, optimizer=self.optimizer, model=self.proj_model, criterion=self.criterion, **kwargs)
            loss_vec.append( loss )

        self.proj_params = proj_params
        
        return loss_vec

    def sample(self, *args, **kwargs):
        print(self.mean.size(), self.subspace.size(), self.proj_params.size())
        map_sample = self.mean + self.subspace.t().matmul(self.proj_params.squeeze(1))

        return map_sample.view(1,-1) 
        
@Inference.register_subclass('vi')
class VI(Inference):

    def __init__(self, base, base_args, base_kwargs, rank, init_inv_softplus_simga=-6.0, prior_log_sigma=0.0):
        super(VI, self).__init__()

        self.vi_model = VIModel(
            base=base,
            base_args=base_args,
            base_kwargs=base_kwargs,
            rank=rank,
            init_inv_softplus_simga=init_inv_softplus_simga,
            prior_log_sigma=prior_log_sigma
        )


    def fit(self, mean, variance, cov_factor, loader, criterion, epochs=100):
        print('Fitting VI')
        self.vi_model.set_subspace(mean, cov_factor)

        elbo = ELBO(criterion, len(loader.dataset))

        optimizer = torch.optim.Adam([param for param in self.vi_model.parameters() if param.requires_grad])

        for _ in range(epochs):
            train_res = train_epoch(loader, self.vi_model, elbo, optimizer)
            print(train_res)



    def sample(self):
        return self.vi_model.sample()

 