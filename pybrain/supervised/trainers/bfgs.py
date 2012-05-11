__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

from scipy import inf, sum, amin, mean, absolute
from scipy.optimize.optimize import fmin_bfgs

from trainer import Trainer


class BFGSTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a
    supervised dataset by backpropagating the errors and calculating weight updates
    using scipy's fmin_bfgs method.
    
    Note that this trainer doesn't have a train() method but only a trainEpochs()
    method, since fmin_bfgs does the iterations internally.
    
    TODO: the callback function should rather be provided to trainEpochs as an 
    argument, than being implemented here.
    """

    def __init__(self, module, ds_train=None, ds_test=None, gtol = 1e-05, norm = inf, 
                 verbose = False, **kwargs):
        """
        Create a BFGSTrainer to train the specified `module` on the
        specified `dataset`.
        """
        Trainer.__init__(self, module)
        self.setData(ds_train)
        self.ds_test = ds_test
        self.verbose = verbose
        self.epoch = 0
        self.totalepochs = 0
        self.train_errors = []
        self.test_errors = []
        self.optimal_params = None
        self.optimal_epoch = 0
        
        self.module = module

    def trainEpochs(self, N):
        """Train the associated module for N epochs."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        
        def updateStatus(params):
            test_error = self.ds_test.evaluateModuleMSE(self.module)
            if self.epoch > 0 and test_error <= amin(self.test_errors):
                self.optimal_params = self.module.params.copy()
                self.optimal_epoch = self.epoch
            print "Epoch %i, E = %g, avg weight: %g" %\
                (self.epoch, (self._last_err / self.ds.getLength()), mean(absolute(self.module.params)))
            print "Test set error: " + str(test_error)
            
            self.train_errors.append(self._last_err / self.ds.getLength())
            self.test_errors.append(test_error)
            self.epoch += 1
            
        def f(params):
            self.module._setParameters(params)
            error = 0
            for seq in self.ds._provideSequences():
                self.module.reset()
                for sample in seq:
                    self.module.activate(sample[0])
                for offset, sample in reversed(list(enumerate(seq))):
                    target = sample[1]
                    outerr = target - self.module.outputbuffer[offset]
                    error += 0.5 * sum(outerr ** 2)
                    
            self._last_err = error
            return error
        
        def df(params):
            self.module._setParameters(params)
            self.module.resetDerivatives()
            for seq in self.ds._provideSequences():
                self.module.reset()
                for sample in seq:
                    self.module.activate(sample[0])
                for offset, sample in reversed(list(enumerate(seq))):
                    target = sample[1]
                    outerr = target - self.module.outputbuffer[offset]
                    str(outerr)
                    self.module.backActivate(outerr)
            # import pdb;pdb.set_trace()
            # self.module.derivs contains the _negative_ gradient
            return -1 * self.module.derivs
        
        new_params = fmin_bfgs(f, self.module.params, df, 
                               maxiter = N, callback = updateStatus, 
                               disp = 0)

        #self.module._setParameters(new_params)
        
        self.epoch += 1
        self.totalepochs += 1
        return self._last_err

    