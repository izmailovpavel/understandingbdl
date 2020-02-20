import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--cls', type=int, required=True, help='class')
parser.add_argument('--scale', type=float, default=5000.)
parser.add_argument('--true_labels', action='store_true')
args = parser.parse_args()

class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

cls = args.cls
print(cls)
nsamples=10000

train_y_file = "train_y.npy" if args.true_labels else "shuffled_train_y.npy"
prefix = "true_" if args.true_labels else "shuffled_true_"
print(train_y_file, prefix)
train_x = np.load("train_x.npy")
train_y = np.load(train_y_file)

mask = (train_y == cls)
train_x = np.concatenate([train_x[mask][:nsamples], train_x[np.logical_not(mask)][:nsamples]], axis=0)
train_y = np.concatenate([train_y[mask][:nsamples], train_y[np.logical_not(mask)][:nsamples]], axis=0)
train_y = (train_y != cls).astype(int)
train_x = torch.from_numpy(train_x) 
train_x = train_x.reshape((train_x.shape[0], -1)).float() / args.scale
train_y = torch.from_numpy(train_y)
print((train_y==0).sum())
print((train_y==1).sum())

model = GPClassificationModel(train_x)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

train_x = train_x.float().cuda()
train_y = train_y.float().cuda()
model = model.cuda()
likelihood = likelihood.cuda()

training_iterations = 100
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    
torch.save(model.state_dict(), '{}model{}.pth'.format(prefix, cls))

train_y = np.load(train_y_file)

train_y = torch.from_numpy(train_y)
train_x = np.load("train_x.npy")
train_x = torch.from_numpy(train_x) 
train_x = train_x.reshape((train_x.shape[0], -1)).float() / args.scale
train_x = train_x.cuda()
train_y = train_y.cuda()

with torch.no_grad():
    for eval_cls in range(10):
        observed_pred = likelihood(model(train_x[train_y == eval_cls]))
        conf = observed_pred.mean.cpu().data.numpy()
        print((conf >= 0.5).mean())
        np.save('{}_conf{}_{}'.format(prefix, cls, eval_cls), conf)

test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")
test_x = torch.from_numpy(test_x).float() / args.scale
test_x = test_x.reshape((test_x.shape[0], -1)).cuda()

with torch.no_grad():
    for eval_cls in range(10):
        observed_pred = likelihood(model(test_x[test_y == eval_cls]))
        conf = observed_pred.mean.cpu().data.numpy()
        print((conf >= 0.5).mean())
        np.save('{}_test_conf{}_{}'.format(prefix, cls, eval_cls), conf)
