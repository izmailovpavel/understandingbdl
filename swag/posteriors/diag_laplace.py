import torch
import numpy as np
import torch.distributions
import time

from ..utils import eval
from ..utils import extract_parameters, set_weights, train_epoch

# def laplace_parameters(module, params, max_num_models=0):
#     for name in list(module._parameters.keys()):
#         if module._parameters[name] is None:
#             print(module, name)
#             continue
#         data = module._parameters[name].data
#         module._parameters.pop(name)
#         module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
#         module.register_buffer('%s_var' % name, data.new(data.size()).zero_())
#         module.register_buffer(name, data.new(data.size()).zero_())
#         params.append((module, name))


class Laplace(torch.nn.Module):
    def __init__(self, base, mean_model, *args, **kwargs):
        super(Laplace, self).__init__()        
        self.params = list()

        self.base_model = base(*args, **kwargs)
        self.mu = torch.cat([p.reshape(-1) for p in mean_model.parameters()])
        self.sigma = torch.zeros_like(self.mu)
        self.rank = self.mu.numel()
        print(self.rank)
        print(sum([p.numel() for p in self.base_model.parameters()]))
        
        set_weights(self.base_model, self.mu, self.mu.device)
        
#         self.base.apply(lambda module: laplace_parameters(module=module, params=self.params))

    def forward(self, input):
        return self.base_model(input)

    def sample(self, scale):
        device = self.mu.device
        w = torch.randn(self.rank, device=self.mu.device) * self.sigma.detach() * scale
        w += self.mu.detach()
        set_weights(self.base_model, w, device)
        return w

#     def export_numpy_params(self):
#         mean_list = []
#         var_list = []
#         for module, name in self.params:
#             mean_list.append(module.__getattr__('%s_mean' % name).cpu().numpy().ravel())
#             var_list.append(module.__getattr__('%s_var' % name).cpu().numpy().ravel())
#         mean = np.concatenate(mean_list)
#         var = np.concatenate(var_list)
#         return mean, var

#     def import_numpy_mean(self, w):
#         k = 0
#         for module, name in self.params:
#             mean = module.__getattr__('%s_mean' % name)
#             s = np.prod(mean.shape)
#             mean.copy_(mean.new_tensor(w[k:k + s].reshape(mean.shape)))
#             k += s

    def estimate_variance(self, loader, criterion, samples=1, prior_var=5e-4):
        
        self.train()
        fisher_diag = torch.zeros_like(self.mu)
        for module, name in self.params:
            var = module.__getattr__('%s_var' % name)
            fisher_diag[(module, name)] = var.new(var.size()).zero_()
        self.sample(scale=0.0)#, require_grad=True)
        
        loss_at_mu = 0.
        ds_size = len(loader.dataset)
        
        for s in range(samples):
            t_s = time.time()
            for input, target in loader:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = self(input)
                distribution = torch.distributions.Categorical(logits=output)
                y = distribution.sample()
                loss = criterion(output, y)
       
                
                loss_at_mu += loss.item() * y.numel() / ds_size / samples
                

                loss.backward()
                
                fisher_diag += torch.cat([p.grad.reshape(-1)**2 for p in self.base_model.parameters()])

            t = time.time() - t_s
            print('%d/%d %.2f sec' % (s + 1, samples, t))

        fisher_diag /= samples
        var = 1.0 / (ds_size * fisher_diag  + prior_var)
        self.sigma = torch.sqrt(var)
        print(loss_at_mu)
        norm_sq = sum([p.norm()**2 for p in self.base_model.parameters()])
        prior_density = -norm_sq  / (2 * prior_var) - np.log(2 * np.pi * prior_var) / 2
        loss_at_mu = loss_at_mu * ds_size - prior_density
        
        marginal_likelihood = np.log(2 * np.pi) * self.rank / 2 
        marginal_likelihood += torch.sum(torch.log(self.sigma)) 
        marginal_likelihood -= loss_at_mu
        return marginal_likelihood

#     def scale_grid_search(self, loader, criterion, logscale_range = torch.arange(-10, 0, 0.5).cuda()):
#         all_losses = torch.zeros_like(logscale_range)
#         t_s = time.time()
#         for i, logscale in enumerate(logscale_range):
#             print('forwards pass with ', logscale)
#             current_scale = torch.exp(logscale)
#             self.sample(scale=current_scale)

#             result = eval(loader, self, criterion)

#             all_losses[i] = result['loss']
        
#         min_index = torch.min(all_losses,dim=0)[1]
#         scale = torch.exp(logscale_range[min_index]).item()
#         t_s_final = time.time() - t_s
#         print('estimating scale took %.2f sec'%(t_s_final))
#         return scale





