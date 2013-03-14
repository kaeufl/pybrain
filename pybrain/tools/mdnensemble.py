__author__ = 'Paul Kaeufl, p.j.kaufl@uu.nl'

import numpy as np

class MDNEnsemble(object):
    def __init__(self, modules=None, dataset=None):
        self.modules=[]
        self.errors=[]
        self.etot=0
        self.ds = dataset
        for m in modules:
            self.addModule(m)
        
    def addModule(self, module):
        if len(self.modules)==0:
            self.M = module.M
            self.c = module.c
            self.indim = module.indim
            self.outdim = module.outdim
        assert module.indim == self.indim, "The given module doesn't match the ensemble. Invalid number of input dimensions."
        assert module.outdim == self.outdim, "The given module doesn't match the ensemble. Invalid number of output dimensions."
        assert module.M == self.M, "The given module doesn't match the ensemble. Invalid number of mixture components."
        assert module.c == self.c, "The given module doesn't match the ensemble. Invalid number of target dimensions."
        self.modules.append(module)
        # evaluate the modules error
        err = 0
        for inp, tgt in self.ds:
            y = module.activate(inp)
            err += module.getError(y, tgt)
        err /= self.ds.getLength()
        self.etot += err 
        self.errors.append(err)
    
    def getPosterior(self, x, t):
        p = np.zeros((len(self.modules), len(t)))
        for k,m in enumerate(self.modules):
            p[k] = m.getPosterior(x, t)
        #import pdb;pdb.set_trace()
        return np.sum(p * np.array(self.errors)[:, None] / self.etot, axis=0)
