__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from supervised import SupervisedDataSet
from pybrain.auxiliary import mdn

class MDNDataSet(SupervisedDataSet):
    def __init__(self, inp, target, M, inp_transform=None, tgt_transform=None):
        SupervisedDataSet.__init__(self, inp, target)
        self.M = M
        self.c = target
        self.inp_transform=inp_transform
        self.tgt_transform=tgt_transform

    def __reduce__(self):
        # not sure why that is overwritten in SequentialDataSet
        # But we have to include additional arguments to __init__ in
        # order to make deepcopy work
        _, _, state, _, _ = super(SupervisedDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.M, self.inp_transform, self.tgt_transform
        return creator, args, state, iter([]), iter({})

    def getTargets(self):
        tgts = []
        for k in range(self.getLength()):
            tgts.append(self.getSample(k)[1])
        return np.array(tgts)