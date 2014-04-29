__author__ = 'Paul Kaeufl, p.j.kaufl@uu.nl'

import numpy as np
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, \
                              BiasUnit, FullConnection

class MixtureDensityMixin(object):
    def _phi(self, T, mu, sigma):
        if T.ndim == 1:
            assert len(T) == mu.shape[-1]
            T = T[None, None,:]
        if T.ndim == 2:
            assert T.shape[1] == mu.shape[-1]
            T = T[None, :, :]
        if mu.ndim == 2:
            mu = mu[None, :, :]
        if sigma.ndim == 1:
            sigma = sigma[None, :]
        dist = np.sum((T[:,:,None,:]-mu[:,None,:,:])**2, axis=-1)
        tmp = np.exp(- 1.0 * dist / (2 * sigma[:,None,:]))
        tmp[tmp < np.finfo('float64').eps] = np.finfo('float64').eps
        tmp *= (1.0 / (2*np.pi*sigma[:,None,:])**(0.5*self.c))
        return np.maximum(tmp, np.finfo(float).eps)
    
    def getPosterior(self, x, t):
        x = np.array(x)
        t = np.array(t)
        if t.ndim == 1:
            if len(t) == self.c:
                t = t[None, None, :]
            elif len(t) == len(x):
                t = t[:, None, None]
            else:
                t = t[None, :, None]
        if t.ndim == 2:
            t = t[None, :, :]
        if x.ndim == 2:
            y = np.zeros((len(x), self.outdim))
            for i,xi in enumerate(x):
                y[i] = self.activate(xi)
        else:
            y = self.activate(x)
        alpha, sigma, mu = self.getMixtureParams(y)
        return self._p(t, alpha, mu, sigma)

    def getError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        #phi = self._phi(t, mu, sigma)
        #tmp = np.maximum(np.sum(alpha * phi, 1), np.finfo(float).eps)
        tmp = np.maximum(self._p(t, alpha, mu, sigma), np.finfo(float).eps)
        return -np.log(tmp)
    
    def getDatasetError(self, dataset):
        Y = self.activateOnDataset(dataset)
        #err = 0
        err = np.sum(self.getError(Y, dataset.getField('target')[:,None,:]))
        #for k,y in enumerate(Y):
        #    err += self.getError(y, dataset.getSample(k)[1])
        return err / dataset.getLength()

class MixtureDensityNetwork(FeedForwardNetwork, MixtureDensityMixin):
    """
    Mixture density network
    """
    def __init__(self, M, c, *args, **kwargs):
        """
        Initialize an MDN with M kernels and output dimension c.
        
                """
        FeedForwardNetwork.__init__(self, *args, **kwargs)
        self.M = M
        self.c = c
        # we have to include the additional arguments in argdict in order to
        # have XML serialization work properly
        self.argdict['c'] = c
        self.argdict['M'] = M
        
    def _p(self, t, alpha, mu, sigma):
        assert t.ndim == 3, "shape of t must be (N, nt, dy)"
        assert alpha.ndim == 2, "shape of alpha must be (N, M)" 
        assert mu.ndim == 3, "shape of mu must be (N, M, dy)"
        assert sigma.ndim == 2, "shape of sigma must be (N, M)"
        phi = self._phi(t, mu, sigma)
        return np.sum(alpha[:,None,:] * phi, axis = -1)
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, None]

    def getMixtureParams(self, y):
        if y.ndim == 1:
            y = y[None, :]
        alpha = np.maximum(self.softmax(y[:, 0:self.M]), np.finfo(float).eps)
        sigma = np.minimum(y[:, self.M:2*self.M], np.log(np.finfo(float).max))
        sigma = np.exp(sigma) # sigma
        sigma = np.maximum(sigma, np.finfo(float).eps)
        mu = np.reshape(y[:, 2*self.M:], (y.shape[0], self.M, self.c))
        return alpha, sigma, mu

    def getOutputError(self, y, t):
        assert y.ndim == 1
        assert t.ndim == 1
        alpha, sigma, mu = self.getMixtureParams(y)
        phi = self._phi(t, mu, sigma)[0,0]
        alpha = alpha[0]
        sigma = sigma[0]
        mu = mu[0]
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
        self.reset() # avoid a memory leak?
        cnet = self.copy()
        net = MixtureDensityNetwork(self.M, self.c)
        self._transferNetStructure(cnet, net)
        net.owner = net
        return net

class PeriodicMixtureDensityNetwork(MixtureDensityNetwork):
    """
    Uses wrapped Gaussian kernels as described 
    in Bishop, Christopher M., and C. Legleye. "Estimating conditional 
    probability densities for periodic variables." (1994): 641-648.
    """
    nperiods = 7
    
    hist_errors = list()
    hist_deriv = list()
    
    def __init__(self, M, c, nperiods, *args, **kwargs):
        MixtureDensityNetwork.__init__(self, M, c, *args, **kwargs)
        self.nperiods = nperiods
        self.argdict['nperiods'] = nperiods
    
    def _p(self, t, alpha, mu, sigma):
        assert t.ndim == 3, "shape of t must be (N, nt, dy)"
        assert alpha.ndim == 2, "shape of alpha must be (N, M)" 
        assert mu.ndim == 3, "shape of mu must be (N, M, dy)"
        assert sigma.ndim == 2, "shape of sigma must be (N, M)"        
        phi = np.zeros([alpha.shape[0], t.shape[1], self.M])
        for l in range(-self.nperiods, self.nperiods+1):
            phi += self._phi(t+l*2*np.pi, mu, sigma)
        return np.sum(alpha[:,None,:] * phi, axis = -1)
    
#    def getError(self, y, t):
#        alpha, sigma, mu = self.getMixtureParams(y)
#        phi = np.zeros([self.M])
#        for l in range(-self.nperiods,self.nperiods+1):
#            phi += self._phi(t+l*2*np.pi, mu, sigma)
#        tmp = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
#        self.hist_errors.append(-np.log(tmp))
#        return -np.log(tmp)
    
    def getOutputError(self, y, t):
        alpha, sigma, mu = self.getMixtureParams(y)
        
        L = np.arange(-self.nperiods, self.nperiods+1)
        chi = t[:, None] + L[None, :]*2*np.pi
        
        Phi = np.zeros([self.M, len(L)])
        for lk in range(len(L)):
            Phi[:, lk] = self._phi(chi[:,lk], mu, sigma)[0,0]
        
        alpha = alpha[0]
        sigma = sigma[0]
        mu = mu[0]

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
        cnet = _PeriodicMixtureDensityNetwork(self.M, self.c, self.nperiods)
        self._transferNetStructure(net, cnet)
        cnet.owner = cnet
        return cnet

    def convertToPythonNetwork(self):
        self.reset() # avoid a memory leak?
        cnet = self.copy()
        net = PeriodicMixtureDensityNetwork(self.M, self.c, self.nperiods)
        self._transferNetStructure(cnet, net)
        net.owner = net
        return net

def buildMixtureDensityNetwork(di, H, dy, M, fast = False, periodic = False,
                               in_ranges=None, out_ranges=None, nperiods=7):
    """
    Construct a Mixture density network. If in_ranges and out_ranges are given a 
    partially connected first layer is created with the given input and output ranges. 
    """
    if type(H) == int:
        H = [H]
    if periodic:
        net = PeriodicMixtureDensityNetwork(M, dy, nperiods)
    else:
        net = MixtureDensityNetwork(M, dy)
    dy = (2+dy)*M
    net.addInputModule(LinearLayer(di, name = 'i'))
    net.addModule(BiasUnit('bias'))
    for i, h in enumerate(H):
        net.addModule(TanhLayer(h, name = 'h%i'%i))
    net.addOutputModule(LinearLayer(dy, name = 'o'))
    if in_ranges and out_ranges:
        # partially connected first layer
        for r_in, r_out in zip(in_ranges, out_ranges):
            net.addConnection(FullConnection(net['i'], net['h0'], 
                                             inSliceFrom=r_in[0],
                                             inSliceTo=r_in[1],
                                             outSliceFrom=r_out[0],
                                             outSliceTo=r_out[1]))
    else:
        # fully connected first layer
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