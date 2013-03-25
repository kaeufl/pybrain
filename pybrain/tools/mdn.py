'''
Created on Mar 19, 2013

@author: kaeufl
'''
import numpy as np
from scipy.special import erf
from pybrain.auxiliary import mdn

def _phi(x):
    tmp = np.exp(-0.5*x**2)
    tmp = max(tmp, np.finfo('float64').eps)
    return 1/np.sqrt(2*np.pi)*tmp

def _Phi(x):
    return 0.5*(1+erf(x/np.sqrt(2)))

def _A(mu, sigma2):
    """
    This is the expectation value of the modulus of a normally distributed 
    random variable.
    """
    mu_sigma = mu / np.sqrt(sigma2)
    tmp = _phi(mu_sigma)
    #if (np.abs(tmp) <= np.finfo('float64').eps):
    #    tmp = np.finfo('float64').eps
    return 2*np.sqrt(sigma2)*tmp + mu*(2*_Phi(mu_sigma)-1)

def crps_gmm(alpha, sigma2, mu, t):
    """
    Evaluate the Continuous Ranked Probability Score (CRPS) for the given
    mixture of Gaussians. The CRPS is a strictly proper scoring rule [1]. Analytic
    formulas for a GMM are given in [2].
    
    References: 
    [1] Tilmann Gneiting and Adrian E. Raftery (2007): Strictly Proper Scoring 
    Rules, Prediction, and Estimation, Journal of the American Statistical 
    Association, 102:477, 359-378
    [2] Grimit, E. P., Gneiting, T., Berrocal, V. J., & Johnson, N. a. (2006). 
    The continuous ranked probability score for circular variables and its 
    application to mesoscale forecast ensemble verification. 
    Quarterly Journal of the Royal Meteorological Society, 132(621C), 2925-2942.
    """
    crps = 0
    for m in range(len(alpha)):
        crps += alpha[m]*_A(t-mu[m], sigma2[m])
        for n in range(len(alpha)):
            crps -= 0.5*alpha[m]*alpha[n]*_A(mu[m]-mu[n], sigma2[m]+sigma2[n])
    return crps

def evalCRPSForSample(net, dataset, sample):
    assert net.c == 1, "CRPS is only implemented for 1-D target vectors."
    y = net.activate(dataset.getSample(sample)[0])
    alpha, sigma2, mu = net.getMixtureParams(y)
    return crps_gmm(alpha, sigma2, mu, dataset.getSample(sample)[1])

def evalCRPSOnDataset(net, dataset, cumulative=True):
    assert net.c == 1, "CRPS is only implemented for 1-D target vectors."
    y = net.activateOnDataset(dataset)
    alpha, sigma2, mu = mdn.getMixtureParams(y, net.M, net.c)
    crps = map(crps_gmm, alpha, sigma2, mu[:,:,0], dataset.getField('target'))
    if cumulative==False:
        return np.array(crps)[:,0]
    return np.sum(crps) / dataset.getLength()