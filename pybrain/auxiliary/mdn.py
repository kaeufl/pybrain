__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np

#def softmax(x):
#    # prevent overflow
#    maxval = np.log(np.finfo(float).max) - np.log(x.shape[0])
#    x = np.minimum(maxval, x)
#    # prevent underflow
#    minval = np.finfo(float).eps
#    x = np.maximum(minval, x)
#    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def softmax(x):
    # prevent overflow
    maxval = np.log(np.finfo(float).max) - np.log(x.shape[1])
    x = np.minimum(maxval, x)
    # prevent underflow
    minval = np.finfo(float).eps
    x = np.maximum(minval, x)
    return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, None]

def getMixtureParams(Y, M, c):
    alpha = np.maximum(softmax(Y[:, 0:M]), np.finfo(float).eps)
    sigma = np.minimum(Y[:, M:2*M], np.log(np.finfo(float).max))
    sigma = np.exp(sigma) # sigma
    sigma = np.maximum(sigma, np.finfo(float).eps)
    mu = np.reshape(Y[:, 2*M:], (Y.shape[0], M, c))
    return alpha, sigma, mu

def mdn_err(y, t, M, c):
    alpha, sigma, mu = getMixtureParams(y, M, c)
    # distance between target data and gaussian kernels
    dist = np.sum((t[None,:]-mu)**2, axis=1)
    phi = (1.0 / (2*np.pi*sigma)**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma))
    phi = np.maximum(phi, np.finfo(float).eps)
    tmp = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
    return -np.log(tmp)

def phi(T, mu, sigma, c):
    dist = np.sum((T[:,None,:]-mu)**2, axis=2)
    phi = (1.0 / (2*np.pi*sigma)**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma))
    return np.maximum(phi, np.finfo(float).eps)