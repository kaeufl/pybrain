__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from supervised import SupervisedDataSet
from pybrain.auxiliary import mdn

class MDNDataSet(SupervisedDataSet):
    def __init__(self, inp, target, M):
        SupervisedDataSet.__init__(self, inp, target)
        self.M = M
        self.c = target

    def __reduce__(self):
        # not sure why that is overwritten in SequentialDataSet
        # But we have to include additional arguments to __init__ in
        # order to make deepcopy work
        _, _, state, _, _ = super(SupervisedDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.M
        return creator, args, state, iter([]), iter({})

    def getTargets(self):
        tgts = []
        for k in range(self.getLength()):
            tgts.append(self.getSample(k)[1])
        return np.array(tgts)