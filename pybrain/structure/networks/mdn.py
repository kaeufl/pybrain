__author__ = 'Paul Kaeufl, p.j.kaufl@uu.nl'

import numpy as np
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, \
                              BiasUnit, FullConnection

class MixtureDensityNetwork(FeedForwardNetwork):
    """
    Mixture density network
    """
    def __init__(self, M, c, *args, **kwargs):
        """
        Initialize an MDN with M kernels and output dimension c.
        
        If periodic is True, wrapped Gaussian kernels are used (e.g. as described 
        in Bishop, Christopher M., and C. Legleye. "Estimating conditional 
        probability densities for periodic variables." (1994): 641-648. )
        """
        FeedForwardNetwork.__init__(self, *args, **kwargs)
        self.M = M
        self.c = c
        # we have to include the additional arguments in argdict in order to
        # have XML serialization work properly
        self.argdict['c'] = c
        self.argdict['M'] = M
        
    def _p(self, t, alpha, mu, sigma):
        phi = self._phi(t, mu, sigma)
        return np.sum(alpha * phi, axis = 1)

    def _phi(self, T, mu, sigma):
        if T.ndim == 1:
            T = T[None,:]
        dist = np.sum((T[:,None,:]-mu[None,:,:])**2, axis=2)
        tmp = np.exp(- 1.0 * dist / (2 * sigma))
        tmp[tmp < np.finfo('float64').eps] = np.finfo('float64').eps
        tmp *= (1.0 / (2*np.pi*sigma)**(0.5*self.c))
        return np.maximum(tmp, np.finfo(float).eps)
    
    def getPosterior(self, x, t):
        y = self.activate(x)
        alpha, sigma, mu = self.getMixtureParams(y)
        return self._p(t, alpha, mu, sigma)

    def softmax(self, x):
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
    
    def _transferNetStructure(self, innet, outnet):
        for m in innet.inmodules:
            outnet.addInputModule(m)
        for m in innet.outmodules:
            outnet.addOutputModule(m)
        for m in innet.modules:
            outnet.addModule(m)

        for clist in innet.connections.values():
            for c in clist:
                outnet.addConnection(c)

        try:
            outnet.sortModules()
        except ValueError:
            print "Network cannot be converted."
            return None
        return True

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
        self._transferNetStructure(net, cnet)
        cnet.owner = cnet
        return cnet

    def convertToPythonNetwork(self):
        cnet = self.copy()
        net = MixtureDensityNetwork(self.M, self.c)
        self._transferNetStructure(cnet, net)
        net.owner = net
        return net

class PeriodicMixtureDensityNetwork(MixtureDensityNetwork):
    nperiods = 7
    
    hist_errors = list()
    hist_deriv = list()
    
    def _p(self, t, alpha, mu, sigma):
        phi = np.zeros([len(t), self.M])
        for l in range(-self.nperiods, self.nperiods+1):
            phi += self._phi(t+l*2*np.pi, mu, sigma)
        return np.sum(alpha * phi, axis = 1)
    
    def getError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        phi = np.zeros([self.M])
        for l in range(-self.nperiods,self.nperiods+1):
            phi += self._phi(t+l*2*np.pi, mu, sigma)
        tmp = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
        self.hist_errors.append(-np.log(tmp))
        return -np.log(tmp)
    
    def getOutputError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        
        L = np.arange(-self.nperiods, self.nperiods+1)
        chi = t[:, None] + L[None, :]*2*np.pi
        
        Phi = np.zeros([self.M, len(L)])
        for lk in range(len(L)):
            Phi[:, lk] = self._phi(chi[:,lk], mu, sigma)
        
        aphi = alpha * np.sum(Phi, axis=1)
        pi = aphi / np.sum(aphi, 0)
        _pi = alpha / np.sum(aphi, 0)

        dE_dy_alpha = alpha - pi
        #dE_dy_sigma = - 0.5 * pi * np.sum(((np.sum((chi-mu[:,:,None])**2, axis=1) / sigma[:,None]) - self.c), axis=1)
        _dE_dy_sigma = Phi * (np.sum((chi[None,:,:] - mu[:,:,None])**2, axis=1) / sigma[:, None] - self.c)
        dE_dy_sigma = -0.5 * _pi * np.sum(_dE_dy_sigma, axis=1)

        _dE_dy_mu = Phi[:,None,:] * ((mu[:,:,None] - chi[None, :, :]) / sigma[:,None,None])
        dE_dy_mu = _pi[:,None] * np.sum(_dE_dy_mu, axis=2)

        outerr = np.zeros(self.outdim)
        outerr[0:self.M] = dE_dy_alpha
        outerr[self.M:2*self.M] = dE_dy_sigma
        outerr[2*self.M:] = np.reshape(dE_dy_mu, (self.M*self.c))
        self.hist_deriv.append(outerr)
        return outerr
    
    def convertToFastNetwork(self):
        """ Attempt to transform the network into a fast network. If fast networks are not available,
        or the network cannot be converted, it returns None. """
        #TODO: this should make use of the base class' convertToFastNetwork
        try:
            from arac.pybrainbridge import _PeriodicMixtureDensityNetwork #@UnresolvedImport
        except:
            print "No fast networks available."
            return None
        net = self.copy()
        cnet = _PeriodicMixtureDensityNetwork(self.M, self.c)
        self._transferNetStructure(net, cnet)
        cnet.owner = cnet
        return cnet

    def convertToPythonNetwork(self):
        cnet = self.copy()
        net = PeriodicMixtureDensityNetwork(self.M, self.c)
        self._transferNetStructure(cnet, net)
        net.owner = net
        return net

def buildMixtureDensityNetwork(di, H, dy, M, fast = False, periodic = False):
    if type(H) == int:
        H = [H]
    if periodic:
        net = PeriodicMixtureDensityNetwork(M, dy)
    else:
        net = MixtureDensityNetwork(M, dy)
    dy = (2+dy)*M
    net.addInputModule(LinearLayer(di, name = 'i'))
    net.addModule(BiasUnit('bias'))
    for i, h in enumerate(H):
        net.addModule(TanhLayer(h, name = 'h%i'%i))
    net.addOutputModule(LinearLayer(dy, name = 'o'))
    net.addConnection(FullConnection(net['i'], net['h0']))
    net.addConnection(FullConnection(net['bias'], net['h0']))
    for i, h in enumerate(H[1:]):
        net.addConnection(FullConnection(net['bias'], net['h%i' % (i+1)]))
        net.addConnection(FullConnection(net['h%i' % (i)], net['h%i' % (i+1)]))    
    net.addConnection(FullConnection(net['bias'], net['o']))
    net.addConnection(FullConnection(net['h%i' % (len(H)-1)], net['o']))
    net.sortModules()
    if fast:
        net = net.convertToFastNetwork()
        net.sortModules()
    return net
