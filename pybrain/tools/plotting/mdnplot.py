__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from pybrain.auxiliary import mdn

class MDNPlotter():
    def __init__(self, module, ds):
        self.module = module
        self.ds = ds
        self.update()
        self.tgts = ds.getTargets()

    def update(self):
        self.y = self.module.activateOnDataset(self.ds)

    def _p(self, t, alpha, mu, sigma):
        return np.sum(alpha * mdn.phi(t, mu, sigma, self.module.c), axis = 1)

    def linTransform(self, x, mu, scale):
        return x*scale+mu

    def plot1DMixture(self, t, alpha, mu, sigma, target = None, linewidth = 2.0):
        if len(t.shape) == 1:
            t = t[:, None]
        p=self._p(t, alpha, mu, sigma)
        plt.plot(t, p, linewidth=linewidth)
        if target:
            plt.vlines(target, np.max(p), np.min(p), linewidth=linewidth)
#            plt.vlines(target, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
#                       linewidth=linewidth)
        return p

    def plot1DMixtureForSample(self, sample, transform=None,
                               show_target_dist=False,
                               show_uniform_prior=False,
                               linewidth=2.0):
        alpha, sigma, mu = self.module.getMixtureParams(self.y[sample])
        tgts = self.tgts
        if transform:
            sigma = sigma*transform['scale']**2
            mu = self.linTransform(mu, transform['mean'], transform['scale'])
            tgts = self.linTransform(tgts, transform['mean'], transform['scale'])
        t = np.linspace(np.min(tgts), np.max(tgts), 150)
        p = self.plot1DMixture(t, alpha, mu, sigma, tgts[sample], linewidth)
        if show_target_dist:
            [h, edges] = np.histogram(tgts, 150, normed=True,
                                      range=(np.min(tgts), np.max(tgts)))
            plt.plot(edges[1:], h, 'g:')
        if show_uniform_prior:
            yprior = 1./(np.max(tgts)-np.min(tgts))
            plt.hlines(yprior, np.min(tgts), np.max(tgts), 'g', linewidth=linewidth)
        #plt.xlim((np.min(tgts), np.max(tgts)))
        #plt.ylim((0, plt.gca().get_ylim()[1]))
        return t, p

    def getPosterior(self, x, t):
        p = []
        if len(t.shape) == 1:
            t = t[:, None]
        if len(x.shape) == 1:
            x = x[:, None]
        for xi in range(len(x)):
            y = self.module.activate(x[xi])
            alpha, sigma, mu = self.module.getMixtureParams(y)
            tmp = self._p(t, alpha, mu, sigma)
            p.append(tmp)
        return np.array(p)

    def plotCenters(self, center = None, transform = None, interactive=False):
        alpha, sigma, mu = mdn.getMixtureParams(self.y, self.module.M, self.module.c)

        tgts = self.tgts
        if transform:
            mu = self.linTransform(mu, transform[1], transform[0])
            tgts = self.linTransform(tgts, transform[1], transform[0])
        if center != None:
            mu = mu[:, center]
        else:
            # select centers with highest mixing coefficient
            maxidxs = np.argmax(alpha, axis=1)
            #print maxidxs
            mu = mu[np.arange(0,len(mu)), maxidxs]

        plt.scatter(mu, tgts, picker=interactive)
        if interactive:
            f=plt.gcf()
            f.canvas.mpl_connect('pick_event', MDNPlotter.scatterPick)
        plt.xlabel('Network prediction')
        plt.ylabel('Target value')
        xlims=[np.min(mu), np.max(mu)]
        plt.xlim(xlims)
        ylims=[np.min(tgts), np.max(tgts)]
        plt.ylim(ylims)

        return mu, self.tgts
    
    @staticmethod
    def scatterPick(event):
        ind = event.ind
        print 'Pattern:', ind