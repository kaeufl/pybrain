__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from supervised import SupervisedDataSet
from arac.cppbridge import SupervisedSimpleDataset

class MDNDataSet(SupervisedDataSet):
    _cdataset = None
    
    def __init__(self, inp, target, M, inp_transform=None, tgt_transform=None):
        """
        Initialize an MDN dataset.
        @param inp: int or np.array, input data or input dimension
        @param target: int or np.array, target data or target dimension
        @param M: number of Gaussian kernels #TODO: remove this from dataset!
        @param inp_transform: set of input transformations that have been
        applied to the dataset #TODO: remove
        @param tgt_transform: #TODO: remove
        """
        SupervisedDataSet.__init__(self, inp, target)
        self.M = M
        if np.isscalar(target):
            self.c = target
        else:
            self.c = target.shape[1]
        self.inp_transform=inp_transform
        self.tgt_transform=tgt_transform

    def __reduce__(self):
        # not sure why this is overwritten in SequentialDataSet
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
    
    def getFastDataset(self):
        if not self._cdataset:
            self._cdataset = SupervisedSimpleDataset(self.indim, self.c)
            for k in range(self.getLength()):
                sample = self.getSample(k)
                self._cdataset.append(sample[0], sample[1])
        return self._cdataset
    
    def appendDataset(self, dataset):
        """
        Append the given dataset.
        """
        for inp, tgt in dataset:
            self.addSample(inp, tgt)
