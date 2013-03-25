__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mpl
from scipy.stats import cumfreq
from pybrain.auxiliary import mdn
from pybrain.structure.networks.mdn import PeriodicMixtureDensityNetwork
#from nnutil.preprocessing import center, standardize, whiten

class MDNPlotter():
    def __init__(self, module, ds):
        self.module = module
        self.ds = ds
        self.y = None
#        self.update()
        self.p = None
        self.tgts = ds.getTargets()

    def update(self):
        self.y = self.module.activateOnDataset(self.ds)
        #self.p = None

    def linTransform(self, x, mu, scale):
        return x*scale+mu

#    def plot1DMixture(self, t, alpha, mu, sigma, target = None, linewidth = 2.0):
#        if t.ndim == 1:
#            t = t[:, None]
#        #p=self._p(t, alpha, mu, sigma)
#        p=self.module._p(t, alpha, mu, sigma)
#        plt.plot(t, p, linewidth=linewidth)
#        if target:
#            plt.vlines(target, np.max(p), 0, linewidth=linewidth)
##            plt.vlines(target, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
##                       linewidth=linewidth)
#        return p
    
    def plot1DMixture(self, x, t, target = None, linewidth = 2.0, 
                      linestyle='k-', transform=None):
        if t.ndim == 1:
            t = t[:, None]
        p = self.module.getPosterior(x, t)
        if transform:
            Dt0 = np.max(t) - np.min(t)
            t = self.linTransform(t, transform['mean'], transform['scale'])
            p /= (np.max(t) - np.min(t))/Dt0
        plt.plot(t, p, linestyle, linewidth=linewidth)
        if target:
            if transform:
                target = self.linTransform(target, transform['mean'], 
                                           transform['scale'])
            plt.vlines(target, np.max(p), 0, linewidth=linewidth)
        return p

#    def plot1DMixtureForSample(self, sample, transform=None,
#                               show_target_dist=False,
#                               show_uniform_prior=False,
#                               prior_range=None,
#                               linewidth=2.0,
#                               res=300,
#                               target_dist_line_style='g:',
#                               target_dist_linewidth=0.5):
#        alpha, sigma, mu = self.module.getMixtureParams(self.y[sample])
#        tgts = self.tgts
#        if transform:
#            sigma = sigma*transform['scale']**2
#            mu = self.linTransform(mu, transform['mean'], transform['scale'])
#            tgts = self.linTransform(tgts, transform['mean'], transform['scale'])
#            if prior_range:
#                prior_range = self.linTransform(prior_range, transform['mean'],
#                        transform['scale'])
#        if prior_range==None:
#            prior_range=[np.min(tgts), np.max(tgts)]
#        
#        t = np.linspace(prior_range[0], prior_range[1], res)
#        #if np.log10(np.max(t)) > 15:
#            
#        p = self.plot1DMixture(t, alpha, mu, sigma, tgts[sample], linewidth)
#            
#        if show_target_dist:
#            [h, edges] = np.histogram(tgts, res, normed=True,
#                                      range=(np.min(tgts), np.max(tgts)))
#            plt.plot(edges[1:], h, target_dist_line_style)
#        if show_uniform_prior:
#            yprior = 1./(prior_range[1]-prior_range[0])
#            plt.hlines(yprior, prior_range[0], prior_range[1], 'g', linewidth=linewidth)
#        #plt.xlim((np.min(tgts), np.max(tgts)))
#        #plt.ylim((0, plt.gca().get_ylim()[1]))
#        return t, p

    def plot1DMixtureForSample(self, sample, transform=None,
                               show_target_dist=False,
                               show_uniform_prior=False,
                               prior_range=None,
                               linewidth=2.0,
                               linestyle='k-',
                               res=300,
                               target_dist_line_style='g:',
                               target_dist_linewidth=0.5):
        tgts = self.tgts
        if prior_range==None:
            prior_range=[np.min(tgts), np.max(tgts)]
        t = np.linspace(prior_range[0], prior_range[1], res)
        smpl = self.ds.getSample(sample)
        p = self.plot1DMixture(smpl[0], t, smpl[1], linewidth, 
                               linestyle=linestyle, 
                               transform=transform)
        if show_target_dist:
            if transform:
                tgts = self.linTransform(tgts, transform['mean'], transform['scale'])
            [h, edges] = np.histogram(tgts, res, normed=True,
                                      range=(np.min(tgts), np.max(tgts)))
            plt.plot(edges[1:], h, target_dist_line_style)
        if show_uniform_prior:
            if transform:
                prior_range = self.linTransform(prior_range, transform['mean'],
                                                transform['scale'])
            yprior = 1./(prior_range[1]-prior_range[0])
            plt.hlines(yprior, prior_range[0], prior_range[1], 'g', linewidth=linewidth)
        
        return t, p

    def plotConditionalForSample(self, sample,
                                 conditional_input_idx=-1,
                                 target_range=None,
                                 res = 300,
                                 c_transform=None,
                                 tgt_transform=None,
                                 pC=None):
        """
        Plot a conditional probability distribution P(y|C). The index of the
        conditional variable can be set with conditional_input_idx and defaults 
        to the last input (-1).
        If pC is a length N array, P(y|C)*P(C)=P(y,C) is plotted instead.
        """
        inputs = self.ds.getField('input')[:,conditional_input_idx]
        targets = self.tgts
        
        if target_range==None:
            target_range=[np.min(targets), np.max(targets)]
        t = np.linspace(target_range[0], target_range[1], res)
        c_range = [np.min(inputs), np.max(inputs)]
        c = np.linspace(c_range[0], c_range[1], res)

        inp = self.ds.getSample(sample)[0]
        tgt = self.ds.getSample(sample)[1]
        
        p = np.zeros([len(c), len(t)])

        for ci in range(len(c)):
            x = inp.copy()
            x[conditional_input_idx] = c[ci]
            p[ci,:] = self.getPosterior(x[None, :], t)[0]

        
        if pC is not None:
            p *= pC[:, None]

        if c_transform is not None:
            tmp = self.linTransform(inputs.copy(), c_transform['mean'], 
                                       c_transform['scale'])
            c_range = [np.min(tmp), np.max(tmp)]
            inp = inp*c_transform['scale'] + c_transform['mean']
            
        if tgt_transform is not None:
            tmp = self.linTransform(targets.copy(), tgt_transform['mean'], 
                                        tgt_transform['scale'])
            target_range=[np.min(tmp), np.max(tmp)]
            tgt = tgt*tgt_transform['scale'] + tgt_transform['mean']
        
        plt.imshow(p.T, origin='lower', 
                   extent=[c_range[0], c_range[1], 
                           target_range[0],target_range[1]],
                   aspect=(c_range[1] - c_range[0]) / (target_range[1] - target_range[0]), 
                   interpolation='none')
        plt.gca().set_xlim([c_range[0], c_range[1]])
        plt.gca().set_ylim([target_range[0], target_range[1]])
        plt.plot(inp[conditional_input_idx], tgt, 'D', 
                 markersize=10,
                 markerfacecolor='red', 
                 markeredgecolor='black', 
                 markeredgewidth=1)
        return p.T
    
    def plotInformationGainDistribution(self):
        dkl = self.getInformationGain()
        plt.hist(dkl, 50)
        plt.xlabel('nats')

#    def getPosterior(self, x, t):
#        if self.p == None:
#            self.p = []
#            if len(t.shape) == 1:
#                t = t[:, None]
#            if len(x.shape) == 1:
#                x = x[:, None]
#            for xi in range(len(x)):
#                y = self.module.activate(x[xi])
#                alpha, sigma, mu = self.module.getMixtureParams(y)
#                #tmp = self._p(t, alpha, mu, sigma)
#                tmp = self.module._p(t, alpha, mu, sigma)
#                self.p.append(tmp)
#            self.p = np.array(self.p)
#        return self.p

    def getPosterior(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        if x.ndim == 1:
            return self.module.getPosterior(x, t)
        elif x.ndim==2 and self.p == None:
            self.p = []
            for xi in range(len(x)):
                self.p.append(self.module.getPosterior(x[xi], t))
            self.p = np.array(self.p)
        return self.p
    
    def plotCenters(self, center = None, transform = None, interactive=False,
                    colors=None, size=20, plot_all_centers=False, square=False,
                    rasterized=False, minimum_gain=None, show_colorbar=False):
        """
        Plot true vs. predicted posterior means.
        @param center:           plot the specified kernel
        @param transform:        apply the specified linear transformation. Transform is
                                 a dict of the form {'mean' : float, 'scale' : float}
        @param interactive:      allow interactive selection of samples by clicking in the plot window
        @param colors:           list of colors: must be of the same length as the number of patterns in the dataset
                                 or one of: 
                                 - 'alpha': the alpha value of the specified kernel is shown on the color-axis
                                 - 'gain': the information gain (kl distance) of the according distribution                                     
        @param size:             size of the dots in scatter plot
        @param plot_all_centers: plot all kernels and plot alpha on the color-axis
        @param minimum_gain:     Discard patterns with an information gain below threshold.
        """
        if self.y == None:
            self.update() 
        alpha, sigma, mu = mdn.getMixtureParams(self.y, self.module.M, self.module.c)
        tgts = self.tgts
        if isinstance(self.module, PeriodicMixtureDensityNetwork):
            while np.any(mu > np.max(tgts)):
                mu[mu > np.max(tgts)] -= 2*np.pi
            while np.any(mu < np.min(tgts)):
                mu[mu < np.min(tgts)] += 2*np.pi
        if transform:
            mu = self.linTransform(mu, transform['mean'], transform['scale'])
            tgts = self.linTransform(tgts, transform['mean'], transform['scale'])
        
        dKL = None
        if minimum_gain:
            dKL = self.getInformationGain()
            idxs = dKL > minimum_gain
            alpha = alpha[idxs]
            sigma = sigma[idxs]
            mu = mu[idxs]
            tgts = tgts[idxs]
            dKL = dKL[idxs]
        
        if center != None:
            mu = mu[:, center]
        elif plot_all_centers==False:
            # select centers with highest mixing coefficient
            #maxidxs = np.argmax(alpha, axis=1)
            maxidxs = self.getMaxKernel(alpha, sigma)
            #print maxidxs
            mu = mu[np.arange(0,len(mu)), maxidxs]

        if plot_all_centers:
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[0.8,.8,.8],[0,0,0]])
            for ctr in range(mu.shape[1]):
                colors=alpha[:, ctr]
                plt.scatter(mu[:, ctr], tgts,  
                            c=colors, 
                            s=size, 
                            cmap=cmap,
                            edgecolor='none',
                            rasterized=rasterized
                            )

        elif colors=='alpha':
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[1.,1.,1.],[0,0,0]])
            colors=alpha[np.arange(0,len(mu)), maxidxs]
            plt.scatter(mu, tgts, picker=interactive, c=colors, s=size, 
                        cmap=cmap, edgecolor='none',
                        rasterized=rasterized
            )
        elif colors=='gain':
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[.8,.8,.8],[0,0,0]])
            if dKL is None:
                dKL = self.getInformationGain()
            plt.scatter(mu, tgts, picker=interactive, c=dKL, s=size, 
                        cmap=cmap, edgecolor='none',
                        rasterized=rasterized
            )
        elif colors is not None:
            plt.scatter(mu, tgts, picker=interactive, c=colors, s=size, 
                        edgecolor='none',
                        rasterized=rasterized
            ) 

        else:
            plt.scatter(mu, tgts, picker=interactive, s=size,
                        rasterized=rasterized
            )
        if interactive:
            f=plt.gcf()
            f.canvas.mpl_connect('pick_event', MDNPlotter.scatterPick)
            
        if show_colorbar:
            plt.colorbar()
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        
        ylims=[np.min(tgts), np.max(tgts)]
        plt.ylim(ylims)
        if not square:
            xlims=[np.min(mu), np.max(mu)]
        else:
            xlims=ylims
        plt.xlim(xlims)
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
        return np.argmax(alpha/np.sqrt(sigma)**self.module.c, axis=1)
        

    def plotRECCurve(self, nbins=20, highlight_error=None, linestyle='-',
                     linewidth=1.0):
        """
        Plot a Regression Error Characteristic (REC) curve.

        The resulting REC curve shows the cumulative distribution of errors
        over the dataset, where the error is measured in distance of the mode
        of the mixture distribution from the target value in standard
        deviations.

        TODO: Use the true mode rather than the kernel with the largest mixing
        coefficient.
        """
        if self.y == None:
            self.update()
        alpha, sigma2, mu = mdn.getMixtureParams(self.y, self.module.M, self.module.c)
        #maxidxs = np.argmax(alpha, axis=1)
        maxidxs=self.getMaxKernel(alpha, sigma2)
        N=len(mu)
        mu = mu[np.arange(0,N), maxidxs]
        sigma2 = sigma2[np.arange(0,N), maxidxs]
        dist = np.sum(np.abs(mu-self.tgts), axis=1)
        dist /= np.sqrt(sigma2)
        h,_,_,_ = cumfreq(dist, nbins,[0,10])
        h/=N
        plt.plot(np.linspace(0,10,nbins), h, linestyle, linewidth=linewidth)
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

    def getInformationGain(self, sample=None, nbins=500):
        """
        Estimate the information gain for every pattern in the dataset. The
        information gain is defined as the Kullback-Leibler divergence between
        prior and posterior distribution.
        """
        eps = np.finfo('float').eps
        prior, t = np.histogram(self.tgts, nbins, density=True)
        dt = np.abs(t[1]-t[0])
        if sample==None:
            posterior = self.getPosterior(self.ds.getField('input'), t)[:,:-1]
        else:
            posterior = self.getPosterior(self.ds.getField('input')[sample], t)[None, :-1]        
        return np.sum(np.where(prior > eps, 
                               posterior * np.log(posterior/(prior+eps)) * dt, 
                               0), 
                      axis=1)

    
