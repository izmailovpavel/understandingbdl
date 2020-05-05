# Bayesian Deep Learning and a Probabilistic Perspective of Generalization

This repository contains experiments for the paper

_Bayesian Deep Learning and a Probabilistic Perspective of Generalization_

by Andrew Gordon Wilson and Pavel Izmailov.

## Introduction

In the paper, we present a probabilistic perspective for reasoning about model construction and generalization, and consider Bayesian deep learning in this context. 
- We show that deep ensembles provide a compelling mechanism for approximate Bayesian inference, and argue that one should think about Bayesian deep learning more from the perspective of integration, rather than simple Monte Carlo, or obtaining precise samples from a posterior.
- We propose MultiSWA and MultiSWAG, which improve over deep ensembles by marginalizing the posterior within multiple basins of attraction.
- We investigate the function-space distribution implied by a Gaussian distribution over weights from multiple different perspectives, considering for example the induced correlation structure across data instances.
- We discuss temperature scaling in Bayesian deep learning.
- We show that results in deep learning that have been presented as mysterious, requiring us to rethink generalization, can naturally be understood from a probabilistic perspective, and can also be reproduced by other models, such as Gaussian processes.
- We argue that while Bayesian neural networks can fit randomly labelled images (which we believe to be a a desirable property), the prior assigns higher mass to structured datasets representative of the problems we want to solve; we discuss this behaviour from a probabilistic perspective and show that Gaussian processes have similar properties.

In this repository we provide code for reproducing results in the paper.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/74967229-9542fc00-53e6-11ea-9b6e-373acf50b185.png" width=500>
</p>

Please cite our work if you find it useful in your research:

```bibtex
@article{wilson2020bayesian,
  title={Bayesian Deep Learning and a Probabilistic Perspective of Generalization},
  author={Wilson, Andrew Gordon and Izmailov, Pavel},
  journal={arXiv preprint arXiv:2002.08791},
  year={2020}
}
```

## Requirements

We use PyTorch 1.3.1 and torchvision 0.4.2 in our experiments. 
Some of the experiments may require other packages: tqdm, numpy, scipy, gpytorch v1.0.0, tabulate, matplotlib, Pillow, wand, skimage, cv2. 

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

To produce the corrupted data use the following script (adapted from [here](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py)).

```bash
python3 ubdl_data/make_cifar_c.py --savepath=<SAVEPATH> --datapath=<DATAPATH>
```
* ```SAVEPATH``` &mdash; path to directory where the data will be saved
* ```DATAPATH``` &mdash; path to directory containing torchvision CIFAR-10

You can then load the data in PyTorch e.g. as follows:
```python
testset = torchvision.datasets.CIFAR10("~/datasets/cifar10/", train=False)
corrupted_testset = np.load("~/datasets/cifar10c/gaussian_noise_5.npz")
testset.data = corrupted_testset["data"]
testset.targets = corrupted_testset["labels"]
```

Below we show an example of images corrupted with _gaussian blur_. 
We also show the negative log likelihood of Deep Ensembles, MultiSWA and MultiSWAG as a function of the number of
independently trained models for different levels of corruption severity.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/74967652-6c6f3680-53e7-11ea-80dd-12e66e3f5ed2.png" width=800>
</p>


## Prior Experiments

For the experiments on prior variance dependence in Lenet-5, VGG-16 and PreResNet-20, we train SWAG models 
with commands listed above, setting `--wd=0.` and
varying `--prior_var` parameter.

For the other experiments on priors we provide iPython notebooks in `experiments/priors`.

In the figure below we show the correlation diagrams between MNIST classes induced by a spherical Gaussian prior on LeNet-5 weights. Left to right: prior std `alpha=0.02`, `alpha=0.1`, `alpha=1.` respectively.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/74967915-e9021500-53e7-11ea-9d60-b99ee17126c5.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/74967916-e9021500-53e7-11ea-9be9-28f0b78bafb1.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/74967919-e99aab80-53e7-11ea-945b-36e524ea30d8.png" width=250>
</p>

## Rethinking Generalization Experiments

The folder `experiments/rethinking_generalization/cifar10_corrupted_labels` contains `.npy` files with
numpy arrays of corrupted CIFAR-10 labels. You can use them with `experiments/train/run_swag.py` using 
`--label_arr <PATH>`, where `<PATH>` is a path to the `.npy` file.

To train Gaussian processes for binary classification on corrupted labels, you can use the script `experiments/rethinking_generalization/gp_train_cifar_binary_corrupted.py`. You can specify the percentage of
altered labels with the `corrupted_labels` argument (e.g. `--corrupted_labels=0.1`).

To train Gaussian processes for one-vs-all classification on corrupted labels, you first need to create the label array
running `python3 experiments/rethinking_generalization/gp_cifar_prepare_data.py`. Then you can run
`python3 experiments/rethinking_generalization/gp_train_cifar_one_vs_all.py --cls=<CLS> [--true_labels]`;
here `<CLS>` is the class for which we train the one-vs-all model, and adding `--true_labels` trains the model on the true
labels instead of corrupted.

Below we show the marginal likelihood approximation (__left__:) for a Gaussian process and (__right__:) PreResNet-20 as a function of the level of label corruption. We use ELBO for GP and Laplace approximation with `swag.posteriors.Laplace` for PreResNet to approximate marginal likelihood.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/74968139-501fc980-53e8-11ea-988f-f702407728e1.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/74968140-501fc980-53e8-11ea-944e-adb722547ff5.png" width=250>
</p>

## Deep Ensembles as BMA

The folder `experiments/deep_ensembles` contains iPython notebooks for the synthetic regression experiment 
connecting deep ensembles and Bayesian model averaging. 
We provide the data used in the experiments as an `.npz` file, the notebook used to generate the data, and
a separate notebook for each baseline.

Below we show the predictive distribution for (__left__:) 200 chains of Hamiltonian Monte Carlo, (__middle__:) deep ensembles and (__right__:) variational inference. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/74969775-2ae08a80-53eb-11ea-9a0e-9b1ce6828c0c.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/74969774-2a47f400-53eb-11ea-9675-abeee0b71a71.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/74969777-2ae08a80-53eb-11ea-97d7-659bb26eb548.png" width=250>
</p>

## Training and Evaluating MultiSWAG

To train a MultiSWAG model you can train several SWAG models independently, and then ensemble the predictions of the samples produced from each of the SWAG models. 
We provide an example script in `experiments/train/run_multiswag.sh`, which trains and evaluates a MultiSWAG model with 3 independent SWAG models using a VGG-16 on CIFAR-100. 


## References for Code Base

This repo was originally forked from [the Subspace Inference GitHub repo](https://github.com/wjmaddox/drbayes).
Code for CIFAR-10-C corruptions is ported from [this GitHub repo](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py).
