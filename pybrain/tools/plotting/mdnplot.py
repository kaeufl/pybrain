__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt

class MDNPlotter():
    def __init__(self, module, trainer):
        self.module = module
        self.trainer = trainer

    def _phi(self, t, mu, sigma):
        # distance between target data and gaussian kernels
        c = self.module.c
        dist = (t[:,None]-mu[None,:])**2
        phi = (1.0 / (2*np.pi*sigma[None, :])**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma[None, :]))
        return np.maximum(phi, np.finfo(float).eps)

    def _p(self, t, alpha, mu, sigma):
        return np.sum(alpha * self._phi(t, mu, sigma), axis = 1)

    def plot1DMixture(self, t, alpha, mu, sigma, target = None):
        p=self._p(t, alpha, mu, sigma)
        plt.plot(t, p)
        if target:
            plt.vlines(target, 0, 1)
        return p

    def getPosterior(self, x, t):
        p = []
        if len(x.shape) == 1:
            x = np.array([x]).T
        for xi in range(len(x)):
            y = self.module.activate(x[xi])
            alpha, sigma, mu = self.trainer.getMixtureParams(y)
            #tmp = np.zeros(len(t))
            tmp = self._p(t, alpha, mu, sigma)
            p.append(tmp)
        return np.array(p)

