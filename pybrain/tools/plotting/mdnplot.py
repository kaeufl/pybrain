__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from pybrain.auxiliary import mdn

class MDNPlotter():
    def __init__(self, module, ds):
        self.module = module
        self.ds = ds
        self.y = self.module.activateOnDataset(ds)
        self.tgts = ds.getTargets()

    def _phi(self, t, mu, sigma):
        # distance between target data and gaussian kernels
        c = self.module.c
        dist = (t[:,None]-mu[None,:])**2
        phi = (1.0 / (2*np.pi*sigma[None, :])**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma[None, :]))
        return np.maximum(phi, np.finfo(float).eps)

    def _p(self, t, alpha, mu, sigma):
        return np.sum(alpha * self._phi(t, mu, sigma), axis = 1)

    def linTransform(self, x, mu, scale):
        return x*scale+mu

    def plot1DMixture(self, t, alpha, mu, sigma, target = None):
        p=self._p(t, alpha, mu, sigma)
        p = p / np.sum(p)
        plt.plot(t, p)
        if target:
            plt.vlines(target, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1])
        return p

    def plot1DMixtureForSample(self, sample, transform=None):
        alpha, sigma, mu = mdn.getMixtureParams(self.y[sample], self.module.M)
        tgts = self.tgts
        if transform:
            sigma = sigma*transform[0]**2
            mu = self.linTransform(mu, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
        t = np.linspace(np.min(tgts), np.max(tgts), 100)
        self.plot1DMixture(t, alpha, mu, sigma, tgts[sample])

    def getPosterior(self, x, t):
        p = []
        if len(x.shape) == 1:
            x = np.array([x]).T
        for xi in range(len(x)):
            y = self.module.activate(x[xi])
            alpha, sigma, mu = mdn.getMixtureParams(y, self.module.M)
            #tmp = np.zeros(len(t))
            tmp = self._p(t, alpha, mu, sigma)
            p.append(tmp)
        return np.array(p)

    def plotCenters(self, center, transform = None):
        centers = []
        for yk in self.y:
            alpha, sigma, mu = mdn.getMixtureParams(yk, self.module.M)
            centers.append(mu)
        centers = np.array(centers)
        tgts = self.tgts
        if transform:
            centers = self.linTransform(centers, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
        plt.scatter(centers[:, center], tgts)
        plt.xlabel('Network prediction')
        plt.ylabel('Target value')
        return centers, self.tgts