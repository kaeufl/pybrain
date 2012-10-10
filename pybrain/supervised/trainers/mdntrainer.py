__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from scg import SCGTrainer
from time import time
from copy import copy
from itertools import islice
from IPython.parallel import Client
from IPython.parallel.util import interactive
from arac.pybrainbridge import _Network
from arac.cppbridge import MDNTrainer as CMDNTrainer

class MDNTrainer(SCGTrainer):
    """Minimise a mixture density error function using a scaled conjugate gradient
    algorithm.
    For details on the function of the Mixture Density Network see Bishop, 1995.

    TODO: This can be considered as a workaround. The error function should really
    not be part of the trainer or the network.
    """
    _cmdntrainer = None
    
    def initNetworkWeights(self, sigma0=1.0, scaled_prior=True):
        """
        initialize weights and biases so that the network models the unconditional density
        of the target data p(t)
        t: target data
        """
        from scipy.cluster.vq import kmeans2, kmeans
        from scipy.spatial.distance import cdist

        t = []
        for k in range(self.ds.getLength()):
            t.append(self.ds.getSample(k)[1])
        t = np.array(t)

        if scaled_prior:
            #self.w1 = np.random.normal(loc=0.0, scale = 1,size=[H, d+1])/np.sqrt(d+1) # 1st layer weights + bias
            #self.w2 = np.random.normal(loc=0.0, scale = 1,size=[ny, H+1])/np.sqrt(H+1) # 2nd layer weights + bias
            sigma1 = 1.0/np.sqrt(self.module.indim+1)
            sigma2 = 1.0/np.sqrt(self.module['h'].indim+1)
        else:
            # init weights from gaussian with width given by prior
            sigma1 = sigma0
            sigma2 = sigma0

        # init connection weights
        conn_i_h = self.module.connections[self.module['i']][0]
        conn_h_o = self.module.connections[self.module['h']][0]
        size_i = conn_i_h.paramdim
        size_h = conn_h_o.paramdim
        conn_i_h.params[:] = np.random.normal(loc=0.0, scale = sigma1,size=size_i)
        conn_h_o.params[:] = np.random.normal(loc=0.0, scale = sigma2,size=size_h)

        # init biases (adapted from netlab, gmminit.m)
        # TODO: here we assume that the first bias connection is always bias->h, the second bias->o
        conn_b_h = self.module.connections[self.module['bias']][0]
        conn_b_o = self.module.connections[self.module['bias']][1]
        
	# added minit="points", since this seems to give better results, with minit="random" kmeans sometimes returns
	# centroids outside target range
        [centroid, label] = kmeans2(t, self.module.M, minit='points')
        cluster_sizes = np.maximum(np.bincount(label), 1) # avoid empty clusters
        alpha = cluster_sizes.astype('float64')/np.sum(cluster_sizes)
        if (self.module.M > 1):
            # estimate variance from the distance to the nearest centre
            sigma = cdist(centroid, centroid)
            sigma = np.min(sigma + np.diag(np.diag(np.ones(sigma.shape))) * 1000, 1)
            sigma = np.maximum(sigma, np.finfo(float).eps) # avoid underflow
        else:
            # only one centre: take average variance
            sigma = [np.mean(np.diag([np.var(t)]))]
        # set biases (adapted from netlab, mdninit.m)
        print "Initial target value distribution"
        print "Alpha:"
        print alpha
        print "Sigma:"
        print sigma
        print "Centers:"
        print centroid
        conn_b_h.params[:] = np.random.normal(loc=0.0, scale = sigma1,size=conn_b_h.paramdim)
        conn_b_o.params[:] = np.reshape([alpha, np.log(sigma), centroid], conn_b_o.params.shape)

    def setData(self, dataset):
        """Associate the given dataset with the trainer."""
        self.ds = dataset
        if dataset:
            assert dataset.indim == self.module.indim

    @staticmethod
    def f(params, trainer):
        #t0=time()
        oldparams=copy(trainer.module.params)
        trainer.module._setParameters(params)
        error = 0
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                y = trainer.module.outputbuffer[offset]
                error += trainer.module.getError(y, target)
        trainer._last_err = error
        trainer.module._setParameters(oldparams)
        #print "f took %.6f" % (time()-t0)
        return error

    @staticmethod
    def df(params, trainer):
        #t0=time()
        oldparams=copy(trainer.module.params)
        trainer.module._setParameters(params)
        trainer.module.resetDerivatives()
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                y = trainer.module.outputbuffer[offset]
                outerr = trainer.module.getOutputError(y, sample[1])
                # str(outerr) # ??? s. backprop trainer
                trainer.module.backActivate(outerr)
        trainer.module._setParameters(oldparams)
        #print "df took %.6f" % (time()-t0)
        return 1 * trainer.module.derivs
    
    def getFastTrainer(self, testset=None):
        assert isinstance(self.module, _Network)
        if not self._cmdntrainer:
            if testset==None:
                self._cmdntrainer = CMDNTrainer(self.module.proxies.map[self.module], 
                                                self.ds.getFastDataset())
            else:
                self._cmdntrainer = CMDNTrainer(self.module.proxies.map[self.module], 
                                                self.ds.getFastDataset(),
                                                testset.getFastDataset())
        return self._cmdntrainer

#    @staticmethod
#    def f(params, trainer):
#        t0=time()
#        oldparams=copy(trainer.module.params)
#        trainer.module._setParameters(params)
#        y = np.zeros([trainer.ds.getLength(), trainer.module.outdim])
#        n = 0
#        for seq in trainer.ds._provideSequences():
#            trainer.module.reset()
#            for offset, sample in list(enumerate(seq)):
#                trainer.module.activate(sample[0])
#                y[n] = trainer.module.outputbuffer[offset]
#            n+=1
#        error=np.sum(trainer.mdn_err(y, trainer.ds.getTargets()))
#        trainer._last_err = error
#        #import pdb;pdb.set_trace()
#        trainer.module._setParameters(oldparams)
#        print "f took %.6f" % (time()-t0)
#        return error

#    @staticmethod
#    def df(params, trainer):
#        t0=time()
#        oldparams=copy(trainer.module.params)
#        trainer.module._setParameters(params)
#        trainer.module.resetDerivatives()
#        N=trainer.ds.getLength()
#        y = np.zeros([N, trainer.module.outdim])
#        n = 0
#        for seq in trainer.ds._provideSequences():
#            trainer.module.reset()
#            for offset, sample in list(enumerate(seq)):
#                trainer.module.activate(sample[0])
#                y[n] = trainer.module.outputbuffer[offset]
#            n+=1
#        tgts=trainer.ds.getTargets()
#        alpha, sigma, mu = trainer.getMixtureParams(y)
#        phi = trainer._phi(tgts, mu, sigma)
#        aphi = alpha*phi
#        pi = aphi / np.sum(aphi, 1)[:, None]
#        dE_dy_alpha = alpha - pi
#        dE_dy_sigma = - 0.5 * pi * ((np.sum((tgts[:,None,:]-mu)**2, axis=2) / sigma) - trainer.c)
#        dE_dy_mu = pi[:,:,None] * (mu - tgts[:,None,:]) / sigma[:,:,None]
#
#        outerr = np.zeros((N, trainer.module.outdim))
#        outerr[:, 0:trainer.M] = dE_dy_alpha
#        outerr[:, trainer.M:2*trainer.M] = dE_dy_sigma
#        outerr[:, 2*trainer.M:] = np.reshape(dE_dy_mu, (N, trainer.M*trainer.c))
#        #str(outerr) # ??? s. backprop trainer
#        n=0
#        for seq in trainer.ds._provideSequences():
#            trainer.module.reset()
#            for sample in seq:
#                trainer.module.activate(sample[0])
#            for offset, sample in reversed(list(enumerate(seq))):
#                trainer.module.backActivate(outerr[n])
#            n+=1
#
#        #print "df took %.6f" % (time()-t0)
#        trainer.module._setParameters(oldparams)
#        print "df took %.6f" % (time()-t0)
#        # note: multiplying by one causes numpy to return a copy instead of a reference
#        return 1*trainer.module.derivs

#    def softmax(self, x):
#        # prevent overflow
#        maxval = np.log(np.finfo(float).max) - np.log(x.shape[1])
#        x = np.minimum(maxval, x)
#        # prevent underflow
#        minval = np.finfo(float).eps
#        x = np.maximum(minval, x)
#        return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, None]

#    def getMixtureParams(self, y):
#        alpha = np.maximum(self.softmax(y[:, 0:self.M]), np.finfo(float).eps)
#        sigma = np.minimum(y[:, self.M:2*self.M], np.log(np.finfo(float).max))
#        sigma = np.exp(sigma) # sigma
#        sigma = np.maximum(sigma, np.finfo(float).eps)
#        mu = np.reshape(y[:, 2*self.M:], (y.shape[0], self.M, self.c))
#        return alpha, sigma, mu
#    def _phi(self, T, mu, sigma):
#        # distance between target data and Gaussian kernels
#        dist = np.sum((T[:,None,:]-mu)**2, axis=2)
#        phi = (1.0 / (2*np.pi*sigma)**(0.5*self.c)) * np.exp(- 1.0 * dist / (2 * sigma))
#        # prevent underflow
#        return np.maximum(phi, np.finfo(float).eps)
#
#    def mdn_err(self, y, t):
#        alpha, sigma, mu = self.getMixtureParams(y)
#        phi = self._phi(t, mu, sigma)
#        tmp = np.maximum(np.sum(alpha * phi, 1), np.finfo(float).eps)
#        return -np.log(tmp)

module = None
sequences = None

class ParallelMDNTrainer(MDNTrainer):
    """
    Trainer performs batch training in parallel on nprocs processors.
    """

    def __init__(self, module, dataset, nprocs, *args, **kwargs):
        self.nprocs = nprocs

        # split dataset into nprocs chunks
        N = dataset.getLength()/nprocs
        print "Training in parallel using %d engines." % nprocs
        self.sequences = list()
        for n in range(nprocs):
            #self.sequences.append(islice(dataset._provideSequences(), n*N, (n+1)*N))
            chunk = list(islice(dataset._provideSequences(), n*N, (n+1)*N))
            self.sequences.append(chunk)
            print "Engine %d: %d samples." % (n, len(chunk))

        self.client = Client()
        dview = self.client[:]

        # XXX: workaround: I don't know of a way to pickle Swig objects. Therefore
        # we convert the C++ network into a python network, push the python network
        # to the engines, and convert it back to a C++ network.
        if isinstance(module, _Network):
            pymodule = module.convertToPythonNetwork()
        else:
            pymodule = module

        print "Pushing training data to engines..."
        for idx, seq in list(enumerate(self.sequences)):
            self.client[idx].push({'sequences': seq})
        print "Done"
        dview.push({'module':pymodule})
        if isinstance(module, _Network):
            dview.apply(ParallelMDNTrainer._convertModule)

        MDNTrainer.__init__(self, module, dataset, *args, **kwargs)

    @staticmethod
    @interactive
    def _convertModule():
        global module
        module=module.convertToFastNetwork()

    @staticmethod
    @interactive
    def _f(params):
        error = 0
        module._setParameters(params)
        #m = module.convertToFastNetwork()
        for seq in sequences:
            module.reset()
            for sample in seq:
                y = module.activate(sample[0])
            #for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                error += module.getError(y, target)
        return error

    @staticmethod
    @interactive
    def _df(params):
        module._setParameters(params)
        module.resetDerivatives()
        for seq in sequences:
            module.reset()
            for sample in seq:
                module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                y = module.outputbuffer[offset]
                outerr = module.getOutputError(y, sample[1])
                # str(outerr) # ??? s. backprop trainer
                module.backActivate(outerr)
        return module.derivs

    @staticmethod
    def f(params, trainer):
        #t0=time()
        #oldparams=copy(trainer.module.params)
        #trainer.module._setParameters(params)
        error = 0
        ar = list()
        for cl in trainer.client:
            ar.append(cl.apply_async(ParallelMDNTrainer._f, params))
        for res in ar:
            error += res.get()

        trainer._last_err = error
        #trainer.module._setParameters(oldparams)
        #print "f took %.6f" % (time()-t0)
        return error

    @staticmethod
    def df(params, trainer):
        #t0=time()
        #oldparams=copy(trainer.module.params)
        #trainer.module._setParameters(params)
        #trainer.module.resetDerivatives()
        ar = list()
        derivs = np.zeros(trainer.module.paramdim)
        for cl in trainer.client:
            ar.append(cl.apply_async(ParallelMDNTrainer._df, params))
        for res in ar:
            derivs += res.get()

        #trainer.module._setParameters(oldparams)
        #print "df took %.6f" % (time()-t0)
        return derivs
