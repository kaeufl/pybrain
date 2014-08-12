__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mpl
from scipy.stats import cumfreq
from scipy.special import erf
from pybrain.auxiliary import mdn
from pybrain.structure.networks.mdn import PeriodicMixtureDensityNetwork
#from nnutil.preprocessing import center, standardize, whiten

class MDNPlotter():
    def __init__(self, module, ds):
        self.module = module
        self.ds = ds
        self.y = None
#        self.update()
        self.p = {}
        self.tgts = ds.getTargets()

    def update(self):
        self.y = self.module.activateOnDataset(self.ds)
        self.p = {}
        #self.p = None

    def linTransform(self, x, mu, scale):
        return x*scale+mu

    def invTransform(self, x, transform):
        if transform.has_key('rescale_params'):
            mean = transform['rescale_params']['mean']
            scale = transform['rescale_params']['scale']
            x = self.linTransform(x, mean, scale)
        if transform.has_key('log') and transform['log'] == True:
            x = np.exp(x)
        return x

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
                      linestyle='-', color='k', transform=None,
                      plot_individual_kernels=False,
                      label = None):
        if t.ndim == 1:
            t = t[:, None]
        t_untransformed = t.copy()
        p = self.module.getPosterior(x, t_untransformed)[0]

        if transform:
            Dt0 = np.max(t) - np.min(t)
            t = self.invTransform(t, transform)
            p /= (np.max(t) - np.min(t))/Dt0
        plt.plot(t, p, linestyle, color=color, linewidth=linewidth, label=label)
        plt.gca().set_ylim(0, 1.1*np.max(p))
        if plot_individual_kernels:
            y = self.module.activate(x)
            alpha, sigma, mu = self.module.getMixtureParams(y)
            for a,s,m in zip(alpha[0], sigma[0], mu[0]):
                pk =  self.module._phi(t_untransformed, m[None,None], s[None,None])
                pk /= (np.max(t) - np.min(t))/Dt0
                plt.plot(t, pk[0,:,0], '--')
        if target:
            if transform:
                target = self.invTransform(target, transform)
            plt.axvline(target, linewidth=linewidth, zorder=10)
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
                               posterior_range=None,
                               linewidth=2.0,
                               linestyle='-',
                               color = 'k',
                               res=300,
                               target_dist_line_style='g:',
                               target_dist_linewidth=0.5,
                               show_target = True,
                               plot_individual_kernels=False,
                               label=None,
                               with_confidence_intervals=False):
        tgts = self.tgts
        if prior_range==None:
            assert len(tgts) > 100, "Dataset too small to estimate target range"
            prior_range=[np.min(tgts), np.max(tgts)]
        if posterior_range is None:
            posterior_range = prior_range
        t = np.linspace(posterior_range[0], posterior_range[1], res)
        smpl = self.ds.getSample(sample)
        target = None
        if show_target:
            target = smpl[1]
        if transform is True:
            transform = self.ds.tgt_transform
        p = self.plot1DMixture(smpl[0], t, target, linewidth, 
                               linestyle=linestyle, 
                               color = color,
                               transform=transform,
                               plot_individual_kernels=plot_individual_kernels,
                               label=label)
        if show_target_dist:
            if transform:
                tgts = self.invTransform(tgts, transform)
            [h, edges] = np.histogram(tgts, res, normed=True,
                                      range=(np.min(tgts), np.max(tgts)))
            plt.plot(edges[1:], h, target_dist_line_style)
        if with_confidence_intervals:
            m0,m1,m2 = self.getMode(sample, res, bool(transform), 
                                    sigma_level=1, 
                                    prior_range=prior_range) 
            plt.axvline(m0, color='g', linestyle='--')
            plt.axvline(m1, color='g', linestyle='--')
            plt.axvline(m2, color='g', linestyle='--')
        if transform:
            t = self.invTransform(t, transform)
            prior_range = self.invTransform(prior_range, transform)
        if show_uniform_prior:
            yprior = np.zeros(len(t))
            yprior[np.logical_and(t >= prior_range[0], t <= prior_range[1])] = 1./(prior_range[1]-prior_range[0])
            #plt.hlines(yprior, prior_range[0], prior_range[1], 'g', linewidth=linewidth)
            plt.plot(t, yprior, 'g', linewidth=linewidth)
            
        plt.gca().set_xlim(t[0], t[-1])
        
        return t, p

    def plotConditionalForSample(self, sample,
                                 conditional_input_idx=-1,
                                 target_range=None,
                                 c_range=None,
                                 res = 300,
                                 c_transform=None,
                                 tgt_transform=None,
                                 pC=None,
                                 plot_target_value=True):
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
        if c_range==None:
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
            #tmp = self.linTransform(inputs.copy(), c_transform['mean'], 
            #                           c_transform['scale'])
            #c_range = [np.min(tmp), np.max(tmp)]
            c_range = self.linTransform(np.array(c_range), c_transform['mean'],
                                        c_transform['scale'])
            inp = inp*c_transform['scale'] + c_transform['mean']
            
        if tgt_transform is not None:
            #tmp = self.linTransform(targets.copy(), tgt_transform['mean'], 
            #                            tgt_transform['scale'])
            #target_range=[np.min(tmp), np.max(tmp)]
#             target_range = self.linTransform(np.array(target_range),
#                                              tgt_transform['mean'], 
#                                              tgt_transform['scale'])
#             tgt = tgt*tgt_transform['scale'] + tgt_transform['mean']
            target_range = self.invTransform(np.array(target_range), tgt_transform)
            tgt = self.invTransform(tgt, tgt_transform)
        plt.imshow(p.T, origin='lower', 
                   extent=[c_range[0], c_range[1], 
                           target_range[0],target_range[1]],
                   aspect=(c_range[1] - c_range[0]) / (target_range[1] - target_range[0]), 
                   interpolation='none')
        plt.gca().set_xlim([c_range[0], c_range[1]])
        plt.gca().set_ylim([target_range[0], target_range[1]])
        if plot_target_value:
            plt.plot(inp[conditional_input_idx], tgt, 'D', 
                     markersize=10,
                     markerfacecolor='red', 
                     markeredgecolor='black', 
                     markeredgewidth=1)
        return p.T
    
    def plotInformationGainDistribution(self, color='k', nbins=50, range=None, normed=False):
        dkl = self.getInformationGain(nbins=nbins)
        plt.hist(dkl, nbins, color=color, range=range, normed=normed)
        plt.xlabel('nats')
        return dkl

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
        idx = hash(str(x)+str(t))
        if not self.p.has_key(idx): 
            self.p[idx] = self.module.getPosterior(x, t)
        return self.p[idx]
    
    def plotCenters(self, center = None, transform = None, interactive=False,
                    colors=None, size=20, plot_all_centers=False, square=False,
                    rasterized=True, minimum_gain=None, show_colorbar=False,
                    edgecolor='none', mode_res=300, gain_nbins=500):
        """
        Plot target value vs. predicted posterior mode.
        @param center:           use the position of the specified kernel, 
                                 if 'max' use the dominant kernel, 
                                 if 'all' plot all kernels
                                 if 'None' (default) use the mode of the distribution
        
        @param transform:        apply the specified linear transformation. Transform is
                                 a dict of the form {'mean' : float, 'scale' : float}
        @param interactive:      allow interactive selection of samples by clicking in the plot window
        @param colors:           list of colors: must be of the same length as the number of patterns in the dataset
                                 or one of: 
                                 - 'alpha': the alpha value of the specified kernel is shown on the color-axis
                                 - 'gain': the information gain (kl distance) of the according distribution                                     
        @param size:             size of the dots in scatter plot
        @param minimum_gain:     Discard patterns with an information gain below threshold.
        @param mode_res:         number of points to be used for mode finding
        """
        if self.y == None:
            self.update()
        tgts = self.tgts
        if center=='max' or center=='all' or minimum_gain or colors=='alpha' or type(center)==int:
            alpha, sigma, mu = mdn.getMixtureParams(self.y, self.module.M, self.module.c)
            if isinstance(self.module, PeriodicMixtureDensityNetwork):
                while np.any(mu > np.max(tgts)):
                    mu[mu > np.max(tgts)] -= 2*np.pi
                while np.any(mu < np.min(tgts)):
                    mu[mu < np.min(tgts)] += 2*np.pi
        if type(center) == int:
            mu = mu[:, center]
        elif center=='max':
            # select centers with highest mixing coefficient
            maxidxs = self.getMaxKernel(alpha, sigma)
            mu = mu[np.arange(0,len(mu)), maxidxs]
        elif center==None:
            mu = self.getMode(res = mode_res)        
        
        if transform:
            mu = self.invTransform(mu, transform)
            tgts = self.invTransform(tgts, transform)
        
        dKL = None
        if minimum_gain:
            dKL = self.getInformationGain(nbins=gain_nbins)
            idxs = dKL > minimum_gain
            alpha = alpha[idxs]
            sigma = sigma[idxs]
            mu = mu[idxs]
            tgts = tgts[idxs]
            dKL = dKL[idxs]
        
        if center=='all':
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[0.8,.8,.8],[0,0,0]])
            for ctr in range(mu.shape[1]):
                colors=alpha[:, ctr]
                plt.scatter(mu[:, ctr], tgts,  
                            c=colors, 
                            s=size, 
                            cmap=cmap,
                            edgecolor=edgecolor,
                            rasterized=rasterized
                            )

        elif colors=='alpha':
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[1.,1.,1.],[0,0,0]])
            colors=alpha[np.arange(0,len(mu)), maxidxs]
            plt.scatter(mu, tgts, picker=interactive, c=colors, s=size, 
                        cmap=cmap, edgecolor=edgecolor,
                        rasterized=rasterized
            )
        elif colors=='gain':
            cmap = mpl.colors.LinearSegmentedColormap.from_list('tmp', 
                                                                [[.8,.8,.8],[0,0,0]])
            #cmap = 'jet'
            if dKL is None:
                dKL = self.getInformationGain(nbins=gain_nbins)
            plt.scatter(mu, tgts, picker=interactive, c=dKL, s=size, 
                        cmap=cmap, edgecolor=edgecolor,
                        rasterized=rasterized
            )
        elif colors is not None:
            plt.scatter(mu, tgts, picker=interactive, c=colors, s=size, 
                        edgecolor=edgecolor,
                        rasterized=rasterized
            ) 

        else:
            plt.scatter(mu, tgts, picker=interactive, s=size,
                        rasterized=rasterized,
                        edgecolor=edgecolor
            )
        if interactive:
            f=plt.gcf()
            f.canvas.mpl_connect('pick_event', MDNPlotter.scatterPick)
            
        if show_colorbar:
            plt.colorbar(use_gridspec=True)
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

    #def getMode(self, alpha, sigma, mu, res=100):
    #    """
    #    Return the mode of the given 1-D mixture.
#
#        The mode is determined numerically using a grid with 'res' points.
#        """
#        # TODO: use getPosterior instead of calling module._p, this avoids many
#        # re-evaluations of the posterior
#        assert mu.ndim == 3, "Wrong number of dimensions"
#        assert mu.shape[2] == 1, "Mode finding is only supported for 1-D mixtures."
#        t0 = np.min(self.ds.getField('target'))
#        t1 = np.max(self.ds.getField('target'))
#         
#        t = np.linspace(t0, t1, res)
#        p = self.module._p(t[None, :, None], alpha, mu, sigma)
#        m = t[np.argmax(p, axis=1)]
#        #m = [t[np.argmax(self.module._p(t[:,None], a, m, s))] for a,m,s in zip(alpha,mu,sigma)]
#        return m

    def _getMode(self, t, p, res=100, transform=False, sigma_level=0):
        m = t[np.argmax(p, axis=1)]
        if sigma_level > 0:
            assert len(p) == 1
            pmax = np.max(p)
            thres = np.exp(-0.5*sigma_level**2)
            tmp = np.where(p[0] <= thres*pmax)[0]
            tmp1 = np.where(tmp - np.argmax(p, axis=1) < 0)[0]
            tmp2 = np.where(tmp - np.argmax(p, axis=1) > 0)[0]
            if len(tmp1):
                m0 = t[tmp[np.max(tmp1)]]
            else:
                m0 = m
            if len(tmp2):
                m1 = t[tmp[np.min(tmp2)]]
            else:
                m1 = m
            m = [m0,m,m1]
        if transform:
            if transform == True:
                transform = self.ds.tgt_transform
            m = self.invTransform(m, transform)
            #m=self.ds.tgt_transform['scale']*m + self.ds.tgt_transform['mean']
        return m
    
    def _getTargetRange(self, prior_range=None, res=100):
        if prior_range is not None:
            t0 = prior_range[0]
            t1 = prior_range[1]
        else:
            assert len(self.ds) > 100, "Dataset too small to estimate target range"
            t0 = np.min(self.ds.getField('target'))
            t1 = np.max(self.ds.getField('target'))
        return np.linspace(t0, t1, res)
    
    def _getCDF(self, t, alpha, sigma2, mu):
        assert mu.shape[-1] == 1, "Multi-dimensional mdns not supported yet"
        cdf = np.zeros((len(alpha), len(t)))
        for m in range(self.module.M):
            x = (t[None,:]-mu[:,m,0][:,None])/np.sqrt(2*sigma2[:,m][:,None])
            cdf += alpha[:,m][:,None]*0.5*(1+erf(x))
        return cdf
    
    def getCDF(self, sample=None, res=100, prior_range=None):
        if sample is not None:
            y = self.module.activate(self.ds.getSample(sample)[0])[None,:]
        else:
            y = self.module.activateOnDataset(self.ds)
        alpha, sigma2, mu = self.module.getMixtureParams(y)
        t = self._getTargetRange(prior_range, res)
        cdf = self._getCDF(t, alpha, sigma2, mu)
        return t, cdf
    
    def getQuantiles(self, sample=None, quantiles=[0.1,0.5,0.9], res=100,
                     prior_range=None):
        t,cdf = self.getCDF(sample, res, prior_range)
        return tuple([t[np.argmax(cdf>p, axis=1)] for p in quantiles])
    
    def getMode(self, sample=None, res=100, transform=False, sigma_level=0,
                prior_range=None):
        """
        Return the mode of 1-D mixture for all samples in the dataset or 
        for the given sample.
        If sigma_level is given, a confidence interval is returned along
        with the mode.

        The mode is determined numerically using a grid with 'res' points.
        """
        t = self._getTargetRange(prior_range, res)
        if sample is None:
            p = self.getPosterior(self.ds.getField('input'), t)
        else:
            p = self.getPosterior(self.ds.getSample(sample)[0], t)
        m = self._getMode(t, p, res, transform, sigma_level)
        return m
    
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
    
    def getRenyiDivergence(self, sample=None, nbins=500):
        """
        Estimate the Renyi divergence between prior and posterior distribution.
        """
        eps = np.finfo('float').eps
        prior, t = np.histogram(self.tgts, nbins, density=True)
        dt = np.abs(t[1]-t[0])
        if sample==None:
            posterior = self.getPosterior(self.ds.getField('input'), t)[:,:-1]
        else:
            posterior = self.getPosterior(self.ds.getField('input')[sample], t)[None, :-1]        
        return np.log(np.sum(np.where(prior > eps, posterior**2 / prior, 0), axis=1))
