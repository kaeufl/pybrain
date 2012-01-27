__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

from scipy import inf, array, sum
from scipy.optimize.optimize import fmin_bfgs

from trainer import Trainer


class BFGSTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a
    supervised dataset by backpropagating the errors and calculating weight updates
    using scipy's fmin_bfgs method.
    """

    def __init__(self, module, dataset=None, gtol = 1e-05, norm = inf, 
                 verbose = False, **kwargs):
        """Create a BFGSTrainer to train the specified `module` on the
        specified `dataset`.
        """
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.verbose = verbose
        self.epoch = 0
        self.totalepochs = 0
        
        self.module = module

    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0
        
        def printStatus(params):
            print "Epoch " + str(self.epoch) + ", E = " +\
                  str(self._last_err / self.ds.getLength())
            
        def f(params):
            self.module._setParameters(params)
            error = 0
            for seq in self.ds._provideSequences():
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
                               maxiter = 1, callback = printStatus, 
                               disp = 0)

        self.module._setParameters(new_params)
        
        self.epoch += 1
        self.totalepochs += 1
        return self._last_err

    