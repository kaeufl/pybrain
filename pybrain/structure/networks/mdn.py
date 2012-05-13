__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, \
                              BiasUnit, FullConnection

class MixtureDensityNetwork(FeedForwardNetwork):
    """Mixture density network
    """
    def __init__(self, H, di, dy, M=1, **args):
        FeedForwardNetwork.__init__(self, **args)
        self.M = M
        self.c = dy
        dy = (2+self.c)*self.M
        self.addInputModule(LinearLayer(di, name = 'i'))
        self.addModule(TanhLayer(H, name = 'h'))
        self.addModule(BiasUnit('bias'))
        self.addOutputModule(LinearLayer(dy, name = 'o'))
        self.addConnection(FullConnection(self['i'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['o']))
        self.addConnection(FullConnection(self['h'], self['o']))
        self.sortModules()

#    def softmax(self, x):
#        # prevent overflow
#        maxval = np.log(np.finfo(float).max) - np.log(x.shape[0])
#        x = np.minimum(maxval, x)
#        # prevent underflow
#        minval = np.finfo(float).eps
#        x = np.maximum(minval, x)
#        return np.exp(x) / np.sum(np.exp(x), axis = 0)
#
#    def getMixtureParams(self, y):
#        alpha = np.maximum(self.softmax(y[0:self.M]), np.finfo(float).eps)
#        sigma = np.minimum(y[self.M:2*self.M], np.log(np.finfo(float).max))
#        sigma = np.exp(sigma) # sigma
#        sigma = np.maximum(sigma, np.finfo(float).eps)
#        mu = np.reshape(y[2*self.M:], [self.c, self.M])
#        return alpha, sigma, mu