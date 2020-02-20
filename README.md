# Bayesian Deep Learning and a Probabilistic Perspective of Generalization

## Requirements

We use PyTorch 1.3.1 and torchvision 0.4.2 in our experiments. Some of the other requirements are listed in `requirements.txt`

All experiments were run on a single GPU.

## File Structure

The files and scripts for reproducing the experiments are organized as follows:

```
.
+-- swag/ (Implements the inference procedures e.g. SWAG, Laplace, SGLD)
+-- ubdl_data/
|   +-- make_cifar_c.py (Script to produce CIFAR-10-C data)
|   +-- corruptions.py (Implements CIFAR-10-C corruptions)
+-- experiments/
|   +-- train/run_swag.py (Script to train SGD, SWAG and SWA models)
|   +-- priors/
|   |   +-- mnist_prior_correlations.ipynb (Prior correlation diagrams)
|   |   +-- cifar-c_prior_correlations.ipynb (Prior correlation structure under perturbations)
|   |   +-- cifar_posterior_predictions.ipynb (Adaptivity of posterior with data and effects of prior variance)
|   |   +-- mnist_prior_samples.ipynb (Visualizing prior sample functions on MNIST)
|   +-- rethinking_generalization/
|   |   +-- cifar10_corrupted_labels/ (Folder with npy arrays of corrupted CIFAR-10 labels)
|   |   +-- gp_train_cifar_one_vs_all.py (Script for training one-vs-all GP models)
|   |   +-- gp_train_cifar_binary_corrupted.py (Script for training binary classification GP models)
|   |   +-- gp_cifar_prepare_data.py (Script to prepare data for one-vs-all models)
|   +-- deep_ensembles/
|   |   +-- 1d regression_data.ipynb (Script used to produce data for deep ensembles as BMA experiment)
|   |   +-- 1d regression_hmc.ipynb (Hamiltonian Monte Carlo)
|   |   +-- 1d regression_deep_ensembles.ipynb (Deep Ensembles)
|   |   +-- 1d regression_svi.ipynb (Variational Inference)
|   |   +-- data.npz (Data saved as an .npz file)
```

## Training SGD, SWAG and SWA models

```bash
# PreResNet20, CIFAR10
# SWAG, SWA:
python experiments/train/run_swag.py --data_path=<DATAPATH> --epochs=300 --dataset=CIFAR10 --save_freq=300 \  
      --model=PreResNet20 --lr_init=0.1 --wd=3e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>
# SGD:
python experiments/train/run_swag.py --data_path=<DATAPATH> --epochs=300 --dataset=CIFAR10 --save_freq=300 \  
      --model=PreResNet20 --lr_init=0.1 --wd=3e-4 --use_test --dir=<DIR>


# VGG16, CIFAR-10
# SWAG:
python experiments/train/run_swag.py --data_path=<DATAPATH> --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>
  
  
# LeNet5, MNIST
# SWAG:
python3 experiments/train/run_swag.py --data_path=~/datasets/ --epochs=50 --dataset=MNIST --save_freq=50  \
      --model=LeNet5 --lr_init=0.05 --swag --swag_start=25 --swag_lr=0.01 --cov_mat --use_test \
      --wd=0. --prior_var=1e-1 --seed 1 --dir=<DIR>
```

## Preparing CIFAR10-C

To produce the corrupted data use the following script.

```bash
python3 ubdl_data/make_cifar_c.py --savepath=<SAVEPATH> --datapath=<DATAPATH>
```
* ```SAVEPATH``` &mdash; path to directory where the data will be saved
* ```DATAPATH``` &mdash; path to directory containing torchvision CIFAR-10

You can then load the data in PyTorch e.g. as follows:
```python
testset = torchvision.datasets.CIFAR10("~/datasets/cifar10/", train=False)
corrupted_testset = np.load("~/datasets/cifar10c/gaussian_noise_5.npz")
```
You can then use `corrupted_testset` as a replacement for the original `testset` dataset.

## Prior Experiments

For the experiments on prior variance dependence in Lenet-5, VGG-16 and PreResNet-20, we train SWAG models 
with comands listed above, setting `--wd=0.` and
varying `--prior_var` parameter.

For the other experiments on priors we provide iPython notebooks in `experiments/priors`.

## Rethinking Generalization Experiments

The folder `experiments/rethinking_generalization/cifar10_corrupted_labels` contains `.npy` files with
numpy arrays of corrupted CIFAR-10 labels. You can use them with `experiments/train/run_swag.py` using 
`--label_arr <PATH>`, where `<PATH>` is a path to the `.npy` file.

To train Gaussian Processes for binary classification on corrupted labels, you can use the script `experiments/rethinking_generalization/gp_train_cifar_binary_corrupted.py`. You can specify the percentage of
altered labels with `----corrupted_labels=0.1`.

To train Gaussian Processes for one-vs-all classification on corrupted labels, you first need to create the label array
running `experiments/rethinking_generalization/gp_cifar_prepare_data.py`. Then you can run
`python3 experiments/rethinking_generalization/gp_train_cifar_one_vs_all.py --cls=<CLS> [--true_labels]`;
here `<CLS>` is the class for which we train the one-vs-all model, and adding `--true_labels` trains the model on the true
labels instead of corrupted.

## Deep Ensembles as BMA

The folder `experiments/deep_ensembles` contains iPython notebooks for the synthetic regression experiment 
connecting deep ensembles and Bayesian model averaging. 
We provide the data used in the experiments as an `.npz` file, the notebook used to generate the data, and
a separate notebook for each baseline.


## References for Code Base

This repo was originally forked from [this GitHub repo](https://github.com/wjmaddox/drbayes).
Code for CIFAR-10-C corruptions is ported from [this GitHub repo](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py)
