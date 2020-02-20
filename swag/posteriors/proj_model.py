import torch
from swag.utils import unflatten_like

class SubspaceModel(torch.nn.Module):
    def __init__(self, mean, cov_factor):
        super(SubspaceModel, self).__init__()
        self.rank = cov_factor.size(0)
        self.register_buffer('mean', mean)
        self.register_buffer('cov_factor', cov_factor)

    def forward(self, t):
        return self.mean + self.cov_factor.t() @ t

class ProjectedModel(torch.nn.Module):
    def __init__(self, proj_params, model, projection = None, mean = None, subspace = None):
        super(ProjectedModel, self).__init__()
        self.model = model

        if subspace is None:
            self.subspace = SubspaceModel(mean, projection)
        else:
            self.subspace = subspace

        if mean is None and subspace is None:
            raise NotImplementedError('Must enter either subspace or mean')

        self.proj_params = proj_params

    def update_params(self, vec, model):
        vec_list = unflatten_like(likeTensorList=list(model.parameters()), vector=vec.view(1,-1))
        for param, v in zip(model.parameters(), vec_list):
            param.detach_()
            param.mul_(0.0).add_(v)

    def forward(self, *args, **kwargs):
        y = self.subspace(self.proj_params)

        self.update_params(y, self.model)
        return self.model(*args, **kwargs)