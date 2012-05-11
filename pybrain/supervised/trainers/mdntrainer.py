__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from scg import SCGTrainer

class MDNTrainer(SCGTrainer):
    """Minimise a mixture density error function using a scaled conjugate gradient
    algorithm.
    For details on the function of the Mixture Density Network see Bishop, 1995.
    
    TODO: This can be considered as a workaround. The error function should really
    not be part of the trainer. Also the number of kernels M should rather be a
    property of the network or the output layer.
    """
    
    def setData(self, dataset):
        """Associate the given dataset with the trainer."""
        self.ds = dataset
        if dataset:
            assert dataset.indim == self.module.indim
    
    def _phi(self, T, mu, sigma, c):
        # distance between target data and gaussian kernels    
        dist = (T-mu)**2
        phi = (1.0 / (2*np.pi*sigma)**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma))
        # prevent underflow
        return np.maximum(phi, np.finfo(float).eps)
    
    def mdn_err(self, y, t):
        alpha, sigma, mu = self.module.getMixtureParams(self.module, y)
        phi = self.module._phi(t, mu, sigma, self.module.c)
        tmp = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
        return -np.log(tmp)
    
    @staticmethod   
    def f(params, trainer):
        trainer.module._setParameters(params)
        error = 0
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                y = trainer.module.outputbuffer[offset]
                error += trainer.mdn_err(y, target)
        trainer._last_err = error
        return error
    
    @staticmethod    
    def df(params, trainer):
        trainer.module._setParameters(params)
        trainer.module.resetDerivatives()
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                y = trainer.module.outputbuffer[offset]
                alpha, sigma, mu = trainer.module.getMixtureParams(y)
                phi = trainer._phi(target, mu, sigma)
                aphi = alpha*phi
                pi = aphi / np.sum(aphi, 0)
                
                dE_dy_alpha = alpha - pi
                dE_dy_sigma = - 0.5 * pi * (((target-mu)**2 / sigma) - trainer.module.c)
                dE_dy_mu = pi * (mu - target) / sigma
                
                outerr = np.zeros(trainer.module.outdim)
                outerr[0:trainer.module.M] = dE_dy_alpha
                outerr[trainer.module.M:2*trainer.module.M] = dE_dy_sigma
                outerr[2*trainer.module.M:] = dE_dy_mu
                str(outerr) # ??? s. backprop trainer
                trainer.module.backActivate(outerr)
        # import pdb;pdb.set_trace()
        # self.module.derivs contains the _negative_ gradient
        return -1 * trainer.module.derivs