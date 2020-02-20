#
# Elliptical slice sampling
#
import math
import numpy as np

def elliptical_slice(initial_theta,prior,lnpdf,
                     cur_lnpdf=None,angle_range=None, **kwargs):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix 
               (like what np.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       kwargs= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,**kwargs)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= np.dot(prior,np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*math.pi
        phi_min= phi-2.*math.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta*math.cos(phi) + nu*math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop,**kwargs)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)

def slice_sample(initial_theta, lnpdf, cur_lnpdf=None,step_out=True, sigma = None, prior=None, **kwargs):
    """
    initial_theta: current value of state
    lnpdf: function to evaluate logp(Data | initial_theta)
    cur_lnpdf: logp(Data |initial_theta)
    step_out: whether to use a stepping out procedure
    sigma: variance hyper-parameter
    from: http://isaacslavitt.com/2013/12/30/metropolis-hastings-and-slice-sampling/
    """
    D = len(initial_theta)
    if sigma is None:
        sigma = np.ones(D) #like in the example
    perm = list(range(D))
    np.random.shuffle(perm)
    #cur_lnpdf = dist.loglike(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf = lnpdf(initial_theta, **kwargs)

    for d in perm:
        llh0 = cur_lnpdf + np.log(np.random.rand())
        rr = np.random.rand(1)
        x_l = initial_theta.copy()
        x_l[d] = x_l[d] - rr * sigma[d]
        x_r = initial_theta.copy()
        x_r[d] = x_r[d] + (1 - rr) * sigma[d]

        if step_out:
            llh_l =lnpdf(x_l, **kwargs)
            while llh_l > llh0:
                x_l[d] = x_l[d] - sigma[d]
                llh_l = lnpdf(x_l)
            llh_r = lnpdf(x_r, **kwargs)
            while llh_r > llh0:
                x_r[d] = x_r[d] + sigma[d]
                llh_r = lnpdf(x_r, **kwargs)

        x_cur = initial_theta.copy()
        while True:
            xd = np.random.rand() * (x_r[d] - x_l[d]) + x_l[d]
            x_cur[d] = xd.copy()
            cur_lnpdf = lnpdf(x_cur, **kwargs)
            # print(cur_lnpdf, llh0)
            if cur_lnpdf > llh0:
                initial_theta[d] = xd.copy()
                break
            elif xd > initial_theta[d]:
                x_r[d] = xd
            elif xd < initial_theta[d]:
                x_l[d] = xd
            else:
                raise RuntimeError('Slice sampler shrank too far.')

    return (initial_theta, cur_lnpdf)