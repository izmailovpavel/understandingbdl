"""
    subspace classes
    CovarianceSpace: covariance subspace
    PCASpace: PCA subspace 
    FreqDirSpace: Frequent Directions Space
"""

import abc

import torch
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition.pca import _assess_dimension_
from sklearn.utils.extmath import randomized_svd


class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            self.subspace = torch.randn(rank, num_parameters)

        if method == 'fastfood':
            raise NotImplementedError("FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def collect_vector(self, vector):
        pass
    
    def get_space(self):
        return self.subspace


@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):

    def __init__(self, num_parameters, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))

        self.max_rank = max_rank

    def collect_vector(self, vector):
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)

    def get_space(self):
        return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()
        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.size()[1]))
        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)


@Subspace.register_subclass('pca')
class PCASpace(CovarianceSpace):

    def __init__(self, num_parameters, pca_rank=20, max_rank=20):
        super(PCASpace, self).__init__(num_parameters, max_rank=max_rank)

        # better phrasing for this condition?
        assert(pca_rank == 'mle' or isinstance(pca_rank, int))
        if pca_rank != 'mle':
            assert 1 <= pca_rank <= max_rank

        self.pca_rank = pca_rank

    def get_space(self):

        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().numpy()

        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.item() - 1))**0.5

        if self.pca_rank == 'mle':
            pca_rank = self.rank.item()
        else:
            pca_rank = self.pca_rank

        pca_rank = max(1, min(pca_rank, self.rank.item()))
        pca_decomp = TruncatedSVD(n_components=pca_rank)
        pca_decomp.fit(cov_mat_sqrt_np)

        _, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)

        # perform post-selection fitting
        if self.pca_rank == 'mle':
            eigs = s ** 2.0
            ll = np.zeros(len(eigs))
            correction = np.zeros(len(eigs))

            # compute minka's PCA marginal log likelihood and the correction term
            for rank in range(len(eigs)):
                # secondary correction term based on the rank of the matrix + degrees of freedom
                m = cov_mat_sqrt_np.shape[1] * rank - rank * (rank + 1) / 2.
                correction[rank] = 0.5 * m * np.log(cov_mat_sqrt_np.shape[0])
                ll[rank] = _assess_dimension_(spectrum=eigs,
                                              rank=rank,
                                              n_features=min(cov_mat_sqrt_np.shape),
                                              n_samples=max(cov_mat_sqrt_np.shape))
            
            self.ll = ll
            self.corrected_ll = ll - correction
            self.pca_rank = np.nanargmax(self.corrected_ll)
            print('PCA Rank is: ', self.pca_rank)
            return torch.FloatTensor(s[:self.pca_rank, None] * Vt[:self.pca_rank, :])
        else:
            return torch.FloatTensor(s[:, None] * Vt)


@Subspace.register_subclass('freq_dir')
class FreqDirSpace(CovarianceSpace):
    def __init__(self, num_parameters, max_rank=20):
        super(FreqDirSpace, self).__init__(num_parameters, max_rank=max_rank)
        self.register_buffer('num_models', torch.zeros(1, dtype=torch.long))
        self.delta = 0.0
        self.normalized = False

    def collect_vector(self, vector):
        if self.rank >= 2 * self.max_rank:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            if s.size >= self.max_rank:
                current_delta = s[self.max_rank - 1] ** 2
                self.delta += current_delta
                s = np.sqrt(s[:self.max_rank - 1] ** 2 - current_delta)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt[:s.size, :])

        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.as_tensor(self.cov_mat_sqrt.size(0))
        self.num_models.add_(1)
        self.normalized = False

    def get_space(self):
        if not self.normalized:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt)
            self.normalized = True
        curr_rank = min(self.rank.item(), self.max_rank)
        return self.cov_mat_sqrt[:curr_rank].clone() / max(1, self.num_models.item() - 1) ** 0.5
