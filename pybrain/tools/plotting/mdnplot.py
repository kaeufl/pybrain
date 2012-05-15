__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import cm
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
        dist = np.sum((t[:,None, None]-mu[None,:, :])**2, axis=2)
        phi = (1.0 / (2*np.pi*sigma[None, :])**(0.5*c)) * np.exp(- 1.0 * dist / (2 * sigma[None, :]))
        return np.maximum(phi, np.finfo(float).eps)

    def _p(self, t, alpha, mu, sigma):
        return np.sum(alpha * self._phi(t, mu, sigma), axis = 1)

    def linTransform(self, x, mu, scale):
        return x*scale+mu

    def plot1DMixture(self, t, alpha, mu, sigma, target = None):
        p=self._p(t, alpha, mu, sigma)
        p = p / np.sum(p)
        plt.plot(t, p, linewidth=1.5)
        ymax = plt.gca().get_ylim()[1]
        if target:
            plt.vlines(target, 0, ymax)
        plt.ylim([0,ymax])
        return p

    def plot1DMixtureForSample(self, sample, transform=None, plot_prior=False):
        alpha, sigma, mu = mdn.getMixtureParams(self.y[sample], self.module.M, self.module.c)
        tgts = self.tgts
        if transform:
            sigma = sigma*transform[0]**2
            mu = self.linTransform(mu, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
        t = np.linspace(np.min(tgts), np.max(tgts), 200)
        self.plot1DMixture(t, alpha, mu, sigma, tgts[sample])
        plt.ylim(0)
        plt.xlim([np.min(tgts), np.max(tgts)])
        if plot_prior:
            yprior = 1./200.
            plt.hlines(yprior, np.min(tgts), np.max(tgts))

    def plot2DMixtureForSample(self, sample,
                                rangex = [-1, 1], rangey = [-1, 1],
                                deltax = 0.01, deltay = 0.01,
                                true_model = True,
                                transform=None):
        M = self.module.M
        alpha, sigma, mu = mdn.getMixtureParams(self.y[sample], M, self.module.c)
        tgts = self.tgts

        if transform:
            sigmax = sigma*transform[0][0,0]**2
            sigmay = sigma*transform[0][0,1]**2
            mu = self.linTransform(mu, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
            rangex = [np.min(tgts[:,0]), np.max(tgts[:,0])]
            rangey = [np.min(tgts[:,1]), np.max(tgts[:,1])]
            deltax = (rangex[1]-rangex[0]) / 200
            deltay = (rangey[1]-rangey[0]) / 200
        else:
            sigmax = sigma
            sigmay = sigma

        print 'mu: ' + str(mu)
        print 'sigma: ' + str(sigmax)

        xlin = np.arange(rangex[0], rangex[1], deltax)
        ylin = np.arange(rangey[0], rangey[1], deltay)
        [XLIN, YLIN] = np.meshgrid(xlin, ylin)

        phi = np.zeros([M,ylin.shape[0], xlin.shape[0]])
        P = np.zeros([ylin.shape[0], xlin.shape[0]])

        for k in range(M):
            phi[k,:,:] = mlab.bivariate_normal(XLIN, YLIN, np.sqrt(sigmax[k]), np.sqrt(sigmay[k]), mu[k,0], mu[k,1])
            P = P + phi[k,:,:] * alpha[k]
        P = P/np.sum(P)
        plt.imshow(P, #interpolation='bilinear',
                    #cmap=cm.gray,
                    origin='lower',
                    extent=[rangex[0],rangex[1],
                            rangey[0],rangey[1]]
                    )
        plt.colorbar(use_gridspec=True)

        #plt.contour(XLIN, YLIN, P,
                                #levels = [0, 1.0/np.exp(1)]

        if true_model:
            #plt.axvline(tgts[sample,0], c = 'r')
            #plt.axhline(tgts[sample,1], c = 'r')
            plt.scatter(tgts[sample,0],tgts[sample,1],marker='*', c="r", s=60)
            print 'true value: ' + str([tgts[sample,0], tgts[sample,1]])

        plt.tight_layout()

    def getPosterior(self, x, t):
        p = []
        if len(x.shape) == 1:
            x = np.array([x]).T
        for xi in range(len(x)):
            y = self.module.activate(x[xi])
            alpha, sigma, mu = mdn.getMixtureParams(y, self.module.M, self.module.c)
            #tmp = np.zeros(len(t))
            tmp = self._p(t, alpha, mu, sigma)
            p.append(tmp)
        return np.array(p)

    def plotCenters(self, center = None, transform = None, dim = 0):
        centers = []
        for yk in self.y:
            alpha, sigma, mu = mdn.getMixtureParams(yk, self.module.M, self.module.c)
            if not center:
                mu = mu[np.argmax(alpha), dim]
            else:
                mu = mu[center, dim]
            centers.append(mu)
        #import pdb;pdb.set_trace()
        centers = np.array(centers)
        tgts = self.tgts[:, dim]
        if transform:
            centers = self.linTransform(centers, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
        plt.scatter(centers, tgts, s=10)
        plt.xlabel('Network prediction')
        plt.ylabel('Target value')
        return centers, self.tgts