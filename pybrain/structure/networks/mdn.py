__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, \
                              BiasUnit, FullConnection

class MixtureDensityNetwork(FeedForwardNetwork):
    """Mixture density network
    """
    def __init__(self, M, c, *args, **kwargs):
        FeedForwardNetwork.__init__(self, *args, **kwargs)
        self.M = M
        self.c = c
        # we have to include the additional arguments in argdict in order to
        # have XML serialization work properly
        self.argdict['c'] = c
        self.argdict['M'] = M

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

    def convertToPythonNetwork(self):
        cnet = self.copy()
        net = MixtureDensityNetwork(self.M, self.c)

        for m in cnet.inmodules:
            net.addInputModule(m)
        for m in cnet.outmodules:
            net.addOutputModule(m)
        for m in cnet.modules:
            net.addModule(m)

        for clist in cnet.connections.values():
            for c in clist:
                net.addConnection(c)

        try:
            net.sortModules()
        except ValueError:
            print "Network cannot be converted."
            return None

        net.owner = net
        return net
    
#class FastMixtureDensityNetwork(MixtureDensityNetwork):
#    """
#    This class implements a hard coded two-layer feed-forward MDN with tanh activation
#    functions, where all calculations are cast as numpy matrix operations for performance.
#    
#    Note 1: It does not make use of any of the PyBrain modular framework, therefore
#    being considered as a quick hack for getting more performance in a special case.
#    
#    Note 2: Not to be confused with the ARAC fast network implementation.
#    """
#    def __init__(self, di, H, dy, M, *args, **kwargs):
#        FeedForwardNetwork.__init__(self, *args, **kwargs)
#        self.M = M
#        self.c = dy
#        self.H = H
#        self.di = di
#        # we have to include the additional arguments in argdict in order to
#        # have XML serialization work properly
#        self.argdict['c'] = dy
#        self.argdict['M'] = M
#        self.argdict['H'] = H
#        self.argdict['di'] = di
#        
#        dy = (2+dy)*M
#        
#        self.paramdim = (di+1)*H+(H+1)*dy
#        
#        # we don't care for mapping the parameters array to some modules here, it's
#        # just a plain simple array
#        # parameters are ordered as follows:
#        # [bias->h, bias->o, i->h, h->o]
#        self.params = np.zeros(self.paramdim)
#        
#        
#    def _setParameters(self, p, owner=None):
#        """ set the parameter array """
#        assert p.shape == [self.paramdim]
#        self.params = p
#        
#    def _forwardImplementation(self, inbuf, outbuf):
#        #self.z = np.append(np.ones([1, x.shape[0]]), self.g(np.dot(w1, x.T)), axis = 0)
#        # calculate output values
#        #y = np.dot(w2, self.z)
#        index = self.H + self.dy
#        x = np.append(np.ones([inbuf.shape[0],1]),x,1)
#        self.z = np.tanh(np.dot(self.params[index:index+self.di], inbuf))
#        z = 
#        
#        
#        assert self.sorted, ".sortModules() has not been called"
#        index = 0
#        offset = self.offset
#        for m in self.inmodules:
#            m.inputbuffer[offset] = inbuf[index:index + m.indim]
#            index += m.indim
#
#        for m in self.modulesSorted:
#            m.forward()
#            for c in self.connections[m]:
#                c.forward()
#
#        index = 0
#        for m in self.outmodules:
#            outbuf[index:index + m.outdim] = m.outputbuffer[offset]
#            index += m.outdim
#
#    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
#        assert self.sorted, ".sortModules() has not been called"
#        index = 0
#        offset = self.offset
#        for m in self.outmodules:
#            m.outputerror[offset] = outerr[index:index + m.outdim]
#            index += m.outdim
#
#        for m in reversed(self.modulesSorted):
#            for c in self.connections[m]:
#                c.backward()
#            m.backward()
#
#        index = 0
#        for m in self.inmodules:
#            inerr[index:index + m.indim] = m.inputerror[offset]
#            index += m.indim

def buildMixtureDensityNetwork(di, H, dy, M, fast = False):
    net = MixtureDensityNetwork(M, dy)
    dy = (2+dy)*M
    net.addInputModule(LinearLayer(di, name = 'i'))
    net.addModule(TanhLayer(H, name = 'h'))
    net.addModule(BiasUnit('bias'))
    net.addOutputModule(LinearLayer(dy, name = 'o'))
    net.addConnection(FullConnection(net['i'], net['h']))
    net.addConnection(FullConnection(net['bias'], net['h']))
    net.addConnection(FullConnection(net['bias'], net['o']))
    net.addConnection(FullConnection(net['h'], net['o']))
    net.sortModules()
    if fast:
        net = net.convertToFastNetwork()
        net.sortModules()
    return net

#def buildFastMixtureDensityNetwork(di, H, dy, M):
#    net = FastMixtureDensityNetwork(di, H, dy, M)
#    #
#    #net.addInputModule(LinearLayer(di, name = 'i'))
#    #net.addModule(TanhLayer(H, name = 'h'))
#    #net.addModule(BiasUnit('bias'))
#    #net.addOutputModule(LinearLayer(dy, name = 'o'))
#    #net.addConnection(FullConnection(net['i'], net['h']))
#    #net.addConnection(FullConnection(net['bias'], net['h']))
#    #net.addConnection(FullConnection(net['bias'], net['o']))
#    #net.addConnection(FullConnection(net['h'], net['o']))
#    #net.sortModules()
#    
#    return net