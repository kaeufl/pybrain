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
    return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, None]

def getMixtureParams(Y, M, c):
    alpha = np.maximum(softmax(Y[:, 0:M]), np.finfo(float).eps)
    sigma = np.minimum(Y[:, M:2*M], np.log(np.finfo(float).max))
    sigma = np.exp(sigma) # sigma
    sigma = np.maximum(sigma, np.finfo(float).eps)
    mu = np.reshape(Y[:, 2*M:], (Y.shape[0], M, c))
    return alpha, sigma, mu

def getError(Y, T, M, c):
    alpha, sigma, mu = getMixtureParams(Y, M, c)
    tmp = phi(T, mu, sigma, c)
    tmp = np.maximum(np.sum(alpha * tmp, 1), np.finfo(float).eps)
    tmp = -np.log(tmp)
    return np.sum(tmp) / Y.shape[0]

def phi(T, mu, sigma, c):
    dist = np.sum((T[:,None,:]-mu)**2, axis=2)
    tmp = np.exp(- 1.0 * dist / (2 * sigma))
    tmp[tmp < np.finfo('float64').eps] = np.finfo('float64').eps
    tmp *= (1.0 / (2*np.pi*sigma)**(0.5*c))
    return np.maximum(tmp, np.finfo(float).eps)
