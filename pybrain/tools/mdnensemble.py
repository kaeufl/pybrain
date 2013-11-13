__author__ = 'Paul Kaeufl, p.j.kaufl@uu.nl'

import numpy as np
from pybrain.structure.networks.mdn import MixtureDensityMixin

class MDNEnsemble(MixtureDensityMixin):
    def __init__(self, modules=None, dataset=None):
        self.modules=[]
        self.errors=[]
        self.etot=0
        self.w=[]
        self.wtot=0
        self.M = 0
        self.indim = 0
        self.outdim = 0
        self.c = 0
        self.module_type = None
        self.ds = dataset
        for m in modules:
            self.addModule(m)
        
    def addModule(self, module):
        # the first module determines input and output dimension
        if len(self.modules)==0:
            self.c = module.c
            self.indim = module.indim
            self._outdim = module.outdim
            self.module_type = type(module)
        assert module.indim == self.indim, "The given module doesn't match the ensemble. Invalid number of input dimensions."
        assert module.outdim == self._outdim, "The given module doesn't match the ensemble. Invalid number of output dimensions."
        assert module.c == self.c, "The given module doesn't match the ensemble. Invalid number of target dimensions."
        assert type(module) == self.module_type
        
        self.modules.append(module)
        self.M += module.M
        self.outdim += module.outdim 
        # evaluate the modules error
        err = 0
        for inp, tgt in self.ds:
            y = module.activate(inp)
            err += module.getError(y, tgt)
        err /= self.ds.getLength()
        self.etot += err 
        self.errors.append(err)
        #self.wtot += 1/err
        #self.w.append(1/err)
        L = np.exp(-err)
        self.wtot += L
        self.w.append(L)
    
    def activate(self, inpt):
        y = np.zeros(self.outdim)
        M = 0
        for m,module in enumerate(self.modules):
            _y = module.activate(inpt)
            y[M:M+module.M] = _y[:module.M]
            y[self.M+M:self.M+M+module.M] = _y[module.M:2*module.M]
            y[2*self.M+self.c*M:2*self.M+self.c*(M+module.M)] = _y[2*module.M:]
            M += module.M
        return y

    def activateOnDataset(self, dataset):
        y = np.zeros((len(dataset), self.outdim))
        for i, sample in enumerate(dataset):
            y[i] = self.activate(sample[0])
        return y

    def getMixtureParams(self, y):
        if y.ndim == 1:
            y = y[None, :]
        alpha = np.zeros((len(y), self.M))
        sigma = np.zeros((len(y), self.M))
        mu = np.zeros((len(y), self.M, self.c))
        M = 0
        for m,module in enumerate(self.modules):
            _y = np.zeros((len(y), module.outdim))
            _y[:, :module.M] = y[:, M:M+module.M]
            _y[:, module.M:2*module.M] = y[:, self.M+M:self.M+M+module.M]
            _y[:, 2*module.M:] = y[:, 2*self.M+self.c*M:2*self.M+self.c*(M+module.M)]
            _alpha, _sigma, _mu = module.getMixtureParams(_y)
            alpha[:, M:M+module.M] = _alpha*self.w[m]/self.wtot
            sigma[:, M:M+module.M] = _sigma
            mu[:, M:M+module.M] = _mu
            M += module.M
        return alpha, sigma, mu
    
    def _p(self, t, alpha, mu, sigma):
        if t.ndim == 1:
            t = t[None, :]
        p = np.zeros((alpha.shape[0], t.shape[1]))
        M = 0
        for _,module in enumerate(self.modules):
            p += module._p(t, alpha[:, M:M+module.M], mu[:, M:M+module.M], 
                    sigma[:, M:M+module.M])
            M += module.M
        return p

    def convertToPythonNetwork(self):
        for k, module in enumerate(self.modules):
            self.modules[k] = module.convertToPythonNetwork()
        return self

    def convertToFastNetwork(self):
        for k, module in enumerate(self.modules):
            self.modules[k] = module.convertToFastNetwork()
        return self

    
