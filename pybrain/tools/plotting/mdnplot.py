__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
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
                               prior_range=None,
                               linewidth=2.0):
        alpha, sigma, mu = self.module.getMixtureParams(self.y[sample])
        tgts = self.tgts
        if transform:
            sigma = sigma*transform['scale']**2
            mu = self.linTransform(mu, transform['mean'], transform['scale'])
            tgts = self.linTransform(tgts, transform['mean'], transform['scale'])
            if prior_range:
                prior_range = self.linTransform(prior_range, transform['mean'],
                        transform['scale'])
        if prior_range==None:
            prior_range=[np.min(tgts), np.max(tgts)]
        t = np.linspace(prior_range[0], prior_range[1], 150)
        p = self.plot1DMixture(t, alpha, mu, sigma, tgts[sample], linewidth)
        if show_target_dist:
            [h, edges] = np.histogram(tgts, 150, normed=True,
                                      range=(np.min(tgts), np.max(tgts)))
            plt.plot(edges[1:], h, 'g:')
        if show_uniform_prior:
            yprior = 1./(prior_range[1]-prior_range[0])
            plt.hlines(yprior, prior_range[0], prior_range[1], 'g', linewidth=linewidth)
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
            #maxidxs = np.argmax(alpha, axis=1)
            maxidxs = self.getMaxKernel(alpha, sigma)
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

    def getMode(self):
        """Return the mode of the distribution for every sample in the
        dataset"""
        pass
    
    def getMaxKernel(self, alpha, sigma):
        """
        Return the mixture component having the largest central value.
        """
        return np.argmax(alpha/sigma**self.module.c, axis=1)
        

    def plotRECCurve(self, nbins=20, highlight_error=None):
        """
        Plot a Regression Error Characteristic (REC) curve.

        The resulting REC curve shows the cumulative distribution of errors
        over the dataset, where the error is measured in distance of the mode
        of the mixture distribution from the target value in standard
        deviations.

        TODO: Use the true mode rather than the kernel with the largest mixing
        coefficient.
        """
        alpha, sigma, mu = mdn.getMixtureParams(self.y, self.module.M, self.module.c)
        #maxidxs = np.argmax(alpha, axis=1)
        maxidxs=self.getMaxKernel(alpha, sigma)
        N=len(mu)
        mu = mu[np.arange(0,N), maxidxs]
        sigma = sigma[np.arange(0,N), maxidxs]
        dist = np.sum(np.abs(mu-self.tgts), axis=1)
        dist /= sigma
        h,_,_,_ = cumfreq(dist, nbins)
        h/=N
        plt.plot(h)
        if highlight_error:
            plt.vlines(highlight_error, 0, 1, linestyles='-.')
        plt.xlabel('$\epsilon$ [n std deviations]')
        plt.ylabel('accuracy')
        return dist
    
    def interactiveScatterPlot(self):
        def updatePlot(pick_event):
            event = pick_event.ind[0]
            print "Selected events: ", pick_event.ind
            print "Plotting event: ", event
            plt.subplot(1,2,2)
            plt.cla()
            mdnplt.plot1DMixtureForSample(event, show_target_dist=True, 
                                          show_uniform_prior=True)

        
        mdnplt = MDNPlotter(self.module, self.ds)
        
        plt.subplot(1,2,1)
        mdnplt.plotCenters()
        f=plt.gcf()
        f.canvas.mpl_connect('pick_event', updatePlot)
