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
        dy = (2+dy)*M
        self.addInputModule(LinearLayer(di, name = 'i'))
        self.addModule(TanhLayer(H, name = 'h'))
        self.addModule(BiasUnit('bias'))
        self.addOutputModule(LinearLayer(dy, name = 'o'))
        self.addConnection(FullConnection(self['i'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['o']))
        self.addConnection(FullConnection(self['h'], self['o']))
        self.sortModules()

    def softmax(self, x):
        # prevent overflow
        maxval = np.log(np.finfo(float).max) - np.log(x.shape[0])
        x = np.minimum(maxval, x)
        # prevent underflow
        minval = np.finfo(float).eps
        x = np.maximum(minval, x)
        return np.exp(x) / np.sum(np.exp(x), axis = 0)

    def getMixtureParams(self, y):
        alpha = np.maximum(self.softmax(y[0:self.M]), np.finfo(float).eps)
        sigma = np.minimum(y[self.M:2*self.M], np.log(np.finfo(float).max))
        sigma = np.exp(sigma) # sigma
        sigma = np.maximum(sigma, np.finfo(float).eps)
        mu = np.reshape(y[2*self.M:], (self.M, self.c))
        return alpha, sigma, mu

    def getError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        phi = self._phi(t, mu, sigma)
        tmp = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
        return -np.log(tmp)

    def _phi(self, T, mu, sigma):
        dist = np.sum((T[None,:]-mu)**2, axis=1)
        phi = (1.0 / (2*np.pi*sigma)**(0.5*self.c)) * np.exp(- 1.0 * dist / (2 * sigma))
        return np.maximum(phi, np.finfo(float).eps)

    def getOutputError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        phi = self._phi(t, mu, sigma)
        aphi = alpha*phi
        pi = aphi / np.sum(aphi, 0)

        dE_dy_alpha = alpha - pi
        dE_dy_sigma = - 0.5 * pi * ((np.sum((t[None,:]-mu)**2, axis=1) / sigma) - self.c)
        #dE_dy_sigma = - 0.5 * pi * (((t-mu)**2 / sigma) - self.c)
        #dE_dy_mu = pi * (mu - t) / sigma
        dE_dy_mu = pi[:,None] * (mu - t[None,:]) / sigma[:,None]

        outerr = np.zeros(self.outdim)
        outerr[0:self.M] = dE_dy_alpha
        outerr[self.M:2*self.M] = dE_dy_sigma
        outerr[2*self.M:] = np.reshape(dE_dy_mu, (self.M*self.c))
        return outerr

    def convertToFastNetwork(self):
        """ Attempt to transform the network into a fast network. If fast networks are not available,
        or the network cannot be converted, it returns None. """
        #TODO: this should make use of the base class' convertToFastNetwork
        try:
            from arac.pybrainbridge import _MixtureDensityNetwork #@UnresolvedImport
        except:
            print "No fast networks available."
            return None

        net = self.copy()

        cnet = _MixtureDensityNetwork(self.M, self.c)

        for m in net.inmodules:
            cnet.addInputModule(m)
        for m in net.outmodules:
            cnet.addOutputModule(m)
        for m in net.modules:
            cnet.addModule(m)

        for clist in net.connections.values():
            for c in clist:
                cnet.addConnection(c)

        try:
            cnet.sortModules()
        except ValueError:
            print "Network cannot be converted."
            return None

        cnet.owner = cnet
        #cnet.M = self.M
        #cnet.c = self.c
        return cnet