__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

from scipy import sum, finfo

from trainer import Trainer
from pybrain.auxiliary.scg import SCG
from copy import copy

class SCGTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a
    supervised dataset by backpropagating the errors and calculating weight updates
    using the scaled conjugate gradient algorithm.
    """

    def __init__(self, module, dataset, totalIterations = 100,
                 xPrecision = finfo(float).eps, fPrecision = finfo(float).eps,
                 init_scg=True, **kwargs):
        """Create a SCGTrainer to train the specified `module` on the
        specified `dataset`.
        """
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.input_sequences = self.ds.getField('input')
        self.epoch = 0
        self.totalepochs = 0
        self.module = module
        #self.tmp_module = module.copy()
        if init_scg:
            self.scg = SCG(self.module.params, self.f, self.df, self,
                           totalIterations, xPrecision, fPrecision,
                           evalFunc = lambda x: str(x / self.ds.getLength()))
        else:
            print "Warning: SCG trainer not initialized!"

    @staticmethod
    def f(params, trainer):
        oldparams=copy(trainer.module.params)
        trainer.module._setParameters(params)
        error = 0
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                outerr = target - trainer.module.outputbuffer[offset]
                error += 0.5 * sum(outerr ** 2)

        trainer._last_err = error
        trainer.module._setParameters(oldparams)
        return error

    @staticmethod
    def df(params, trainer):
        oldparams=copy(trainer.module.params)
        trainer.module._setParameters(params)
        trainer.module.resetDerivatives()
        for seq in trainer.ds._provideSequences():
            trainer.module.reset()
            for sample in seq:
                trainer.module.activate(sample[0])
            for offset, sample in reversed(list(enumerate(seq))):
                target = sample[1]
                outerr = target - trainer.module.outputbuffer[offset]
                str(outerr)
                trainer.module.backActivate(outerr)
        # import pdb;pdb.set_trace()
        trainer.module._setParameters(oldparams)
        # self.module.derivs contains the _negative_ gradient?
        return -1 * trainer.module.derivs

    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()

        # run SCG for one epoch
        new_params = self.scg.scg(self.module.params)
        self.module._setParameters(new_params['x'])

        print "Epoch " + str(self.epoch) + ", E = " +\
              str(self._last_err / self.ds.getLength()) +\
              ", beta = " + str(self.scg.beta)

        self.epoch += 1
        self.totalepochs += 1
        return self._last_err / self.ds.getLength()
