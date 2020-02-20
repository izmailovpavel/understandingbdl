import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

from swag import data, models, utils, losses
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--corrupted_labels', type=float, required=True, 
                    help='% of corrupted labels')
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


loaders, num_classes = data.loaders(
    "CIFAR10",
    "~/datasets/",
    1000,
    4,
    None,
    None,
    use_validation=False,
    split_classes=None,
    shuffle_train=False
)

train_x = loaders['train'].dataset.data
train_y = np.array(loaders['train'].dataset.targets)

mask = np.logical_or(train_y==0, train_y==1) # airplane and car
train_x = train_x[mask]
train_y = train_y[mask]

corrupted_labels = int(len(train_y) * args.corrupted_labels)
idx = np.arange(len(train_y))
np.random.shuffle(idx)
idx = idx[:corrupted_labels]
print("{} labels corrupted".format(len(idx)))
train_y[idx] = 1-train_y[idx]

train_x = torch.from_numpy(train_x) 
train_x = train_x.reshape((train_x.shape[0], -1)).float() / 5000
train_y = torch.from_numpy(train_y)

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
    
torch.save(model.state_dict(), 'bin_model_corrupted{}.pth'.format(args.corrupted_labels))

model.eval()
likelihood.eval()

with torch.no_grad():
    output = model(train_x)
    observed_pred = likelihood(output)
    preds = observed_pred.mean.ge(0.5).float()
    print("Train Accuracy", (preds == train_y).float().mean())
    print("MLL:", mll(output, train_y))

    test_x = loaders['test'].dataset.data
    test_y = np.array(loaders['test'].dataset.targets)
    mask = np.logical_or(test_y==0, test_y==1)
    test_x = test_x[mask]
    test_y = test_y[mask]
    test_x = torch.from_numpy(test_x).float() / 5000
    test_x = test_x.reshape((test_x.shape[0], -1)).cuda()
    observed_pred = likelihood(model(test_x))
    preds = observed_pred.mean.ge(0.5).float().cpu().numpy()
    print("Test accuracy:", (preds == test_y).mean())
