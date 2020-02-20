import torch
import numpy as np

from .elliptical_slice import elliptical_slice, slice_sample
from .proj_model import ProjectedModel

class EllipticalSliceSampling(torch.nn.Module):
    def __init__(self, base, subspace, var, loader, criterion, num_samples = 20, 
        use_cuda = False, method='elliptical', *args, **kwargs):
        super(EllipticalSliceSampling, self).__init__()

        if method=='elliptical':
            self.slice_method = elliptical_slice
        if method=='slice':
            self.slice_method = slice_sample

        self.base_model = base(*args, **kwargs)
        if use_cuda:
            self.base_model.cuda()

        self.base_params = []
        for name, param in self.base_model.named_parameters():
            self.base_params.append([param, name, param.size()])

        self.subspace = subspace
        self.var = var
        
        self.loader = loader
        self.criterion = criterion

        self.num_samples = num_samples
        self.use_cuda = use_cuda

        self.all_samples = None

        self.model = self.base_model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def prior_sample(self, prior='identity', scale=1.0):
        if prior=='identity':
            cov_mat = np.eye(self.subspace.cov_factor.size(0))

        elif prior=='schur':
            trans_cov_mat = self.subspace.cov_factor.matmul(self.subspace.cov_factor.subspace.t()).numpy()
            trans_cov_mat /= (self.swag_model.n_models.item() - 1)
            cov_mat = np.eye(self.subspace.cov_factor.size(0)) + trans_cov_mat

        else:
            raise NotImplementedError('Only schur and identity priors have been implemented')

        cov_mat *= scale
        sample = np.random.multivariate_normal(np.zeros(self.subspace.cov_factor.size(0)), cov_mat.astype(np.float64), 1)[0,:]
        return sample

    def log_pdf(self, params, temperature = 1., minibatch = False):
        params_tensor = torch.FloatTensor(params)
        params_tensor = params_tensor.view(-1)

        if self.use_cuda:
            params_tensor = params_tensor.cuda()
        with torch.no_grad():
            proj_model = ProjectedModel(model=self.base_model, subspace = self.subspace, proj_params = params_tensor)
            loss = 0
            num_datapoints = 0.0
            for batch_num, (data, target) in enumerate(self.loader):
                if minibatch and batch_num > 0:
                    break
                num_datapoints += data.size(0)
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                batch_loss, _, _ = self.criterion(proj_model, data, target)
                loss += batch_loss

        loss = loss / (batch_num+1) * num_datapoints
        return -loss.cpu().numpy() / temperature

    def fit(self, use_cuda = True, prior='identity', scale=1.0, **kwargs):
        # initialize at prior mean = 0
        current_sample = np.zeros(self.subspace.cov_factor.size(0))
        
        all_samples = np.zeros((current_sample.size, self.num_samples))
        logprobs = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            prior_sample = self.prior_sample(prior=prior, scale=scale)
            current_sample, logprobs[i] = self.slice_method(initial_theta=current_sample, prior=prior_sample, 
                                                lnpdf=self.log_pdf,  **kwargs)
            # print(logprobs[i])
            all_samples[:,i] = current_sample
        
        self.all_samples = all_samples
        return logprobs

    def sample(self, ind=None, *args, **kwargs):
        if ind is None:
            ind = np.random.randint(self.num_samples)
        
        rsample = torch.FloatTensor(self.all_samples[:,int(ind)])

        #sample = self.subspace(torch.FloatTensor(rsample)).view(-1)

        if self.use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        rsample = rsample.to(device)

        self.model = ProjectedModel(model=self.base_model, subspace=self.subspace, proj_params=rsample)
        return rsample

        
        

