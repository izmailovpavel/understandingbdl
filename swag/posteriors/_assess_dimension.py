import numpy as np

from scipy.special import gammaln 

def _assess_dimension_(spectrum, unscaled_vhat, rank, n_samples, n_features, alpha = 1, beta = 1):
    """Compute the likelihood of a rank ``rank`` dataset
    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.
    Parameters
    ----------
    spectrum : array of shape (n)
        Data spectrum.
    rank : int
        Tested rank value.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    Returns
    -------
    ll : float,
        The log-likelihood
    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """
    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the"
                         " dataset")

    pu = -rank * np.log(2.)
    for i in range(rank):
        pu += (gammaln((n_features - i) / 2.) -
               np.log(np.pi) * (n_features - i) / 2.)
    #pu -= rank * gammaln(alpha/2) + gammaln(alpha * (n_features - rank)/2)
    #pu += alpha * (n_features - rank) / 2 * np.log(beta * (n_features - rank) / 2) + alpha * rank * np.log(beta / 2)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    if rank == n_features:
        pv = 0
        v = 1
    else:
        #v = np.sum(spectrum[rank:]) / (n_features - rank)
        v = unscaled_vhat / (n_features - rank)
        #print(-np.log(v))
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = np.log(2. * np.pi) * (m + rank + 1.) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            #print((spectrum[i] - spectrum[j]) *
            #          (1. / spectrum_[j] - 1. / spectrum_[i]), i, j)
            pa += np.log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + np.log(n_samples)

    #print(pu, pl-pa/2., pv, pp)
    #pu = 0
    #pp = 0
    #pa = 0
    #pv = 0
    ll = pu + pl + pv + pp - pa / 2. - (rank + m) * np.log(n_samples) * 3 / 2.
    return ll

def _infer_dimension_(spectrum, tr_sigma, n_samples, n_features, delta = None, alpha = 1, beta = 1):
    """Infers the dimension of a dataset of shape (n_samples, n_features)
    The dataset is described by its spectrum `spectrum`.
    """
    n_spectrum = len(spectrum)
    ll = np.empty(n_spectrum)
    unscaled_vhat = np.empty(n_spectrum)
    for rank in range(n_spectrum):
        if delta is not None:
            unscaled_vhat[rank] = tr_sigma - (rank * delta / (n_samples - 1) + spectrum[:rank].sum())
            #print('unscaled_vhat is : ', unscaled_vhat)
        else:
            unscaled_vhat[rank] = tr_sigma - spectrum[:rank].sum()

        ll[rank] = _assess_dimension_(spectrum, unscaled_vhat[rank], rank, n_samples, n_features, alpha = alpha, beta = beta)
    return np.nanargmax(ll)+1, ll, unscaled_vhat