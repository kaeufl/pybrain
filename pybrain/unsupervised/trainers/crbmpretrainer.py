'''
Created on Feb 5, 2013

@author: Paul Kaeufl (p.j.kaufl@uu.nl)
'''
import numpy as np
import autoencoder
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import Trainer
from pybrain.structure.modules import TanhLayer, BiasUnit
from pybrain.structure.modules.neuronlayer import NeuronLayer

class CrbmPreTrainer(Trainer):
    """
    Performs greedy layer-wise pre-training using a continuous restricted 
    Boltzmann machine (see H.Chen and A.F. Murray, 2003). 
    
    Note: This module requires a proprietary autoencoder C library, which is 
    not part of this distribution.
    """
    def __init__(self, module, dataset = None, init_std = 0.01, 
                 noise_std = 0.1, learning_rate = 0.3):
        for m in module.modules:
            if m in module.inmodules or m in module.outmodules:
                continue
            if isinstance(m, NeuronLayer) and not isinstance(m, BiasUnit) \
               and not isinstance(m, TanhLayer):
                raise ValueError("Only tanh hidden layers are supported.")
        self.setData(dataset)
        self.init_std = init_std
        self.noise_std = noise_std
        self.lrate = learning_rate
        self._crbms = []
        self._modules = []
        Trainer.__init__(self, module)
        
    def setData(self, dataset):
        """Associate the given dataset with the trainer."""
        if dataset and not isinstance(dataset, SupervisedDataSet):
            raise ValueError("Only supervised datasets are supported.")
        self.ds = dataset
        if dataset:
            assert dataset.indim == self.module.indim
            data = dataset.getField('input')
            scale = np.ones((len(data)))
            weight = np.ones((len(data)))
            self._autoencoder_ds = autoencoder.Dataset(data, scale, weight)
        
    def initCrbms(self):
        layers = [i for i in self.module.modulesSorted 
                  if isinstance(i, TanhLayer) or i in self.module.inmodules]
        self._modules = zip(layers, layers[1:])
        bias_unit = [i for i in self.module.modulesSorted if isinstance(i, BiasUnit)][0]
        self._biascons = [c for c in self.module.connections[bias_unit] 
                         if c.outmod not in self.module.outmodules] 
        self._biascons.sort(key=lambda c: layers.index(c.outmod))
        for visible, hidden in self._modules:
            self._crbms.append(self.getCrbmFromModules(visible, hidden))
            
    
    def getCrbmFromModules(self, visible, hidden):
        crbm = autoencoder.CRBM(visible.dim, hidden.dim, self.init_std, -1.0, 
                                1.0) 
        return crbm
        
    def trainEpochs(self, epochs=1, verbose=True):
        if not self._autoencoder_ds:
            raise Exception("No dataset provided.")
        if not self._crbms:
            self.initCrbms()
        
        dataset = self._autoencoder_ds
        for k, crbm in enumerate(self._crbms):
            print "Pre-training layer %i for %i epochs..." % (k+1, epochs)
            crbm.train(dataset, epochs, self.noise_std, self.lrate,
                       verbose)
            if verbose:
                w = np.abs(crbm.getW())
                b_v2h = np.abs(crbm.getB_v2h())
                b_h2v = np.abs(crbm.getB_h2v())
                a_v2h = np.abs(crbm.getA_v2h())
                a_h2v = np.abs(crbm.getA_h2v())
                print "Max/Min/Avg absolute weight:             %f / %f / %f" % (np.max(w), np.min(w), np.mean(w))
                print "Max/Min/Avg absolute bias (v->h):        %f / %f / %f" % (np.max(b_v2h), np.min(b_v2h), np.mean(b_v2h))
                print "Max/Min/Avg absolute bias (h->v):        %f / %f / %f" % (np.max(b_h2v), np.min(b_h2v), np.mean(b_h2v))
                print "Max/Min/Avg absolute sensitivity (v->h): %f / %f / %f" % (np.max(a_v2h), np.min(a_v2h), np.mean(a_v2h))
                print "Max/Min/Avg absolute sensitivity (h->v): %f / %f / %f" % (np.max(a_h2v), np.min(a_h2v), np.mean(a_h2v))
            print "Done."
            # update network weights and biases
            conn_v2h = self.module.connections[self._modules[k][0]][0]
            w = crbm.getW()
            conn_v2h.params[:] = (w * crbm.getA_v2h()[None, :]).T.reshape(w.shape[0] * w.shape[1])
            self._biascons[k].params[:] = crbm.getB_v2h() * crbm.getA_v2h()
            
            # new dataset is formed by encoding the current dataset
            dataset = crbm.encode(dataset)

    def train(self):
        """Train on the current dataset, for a single epoch."""
        return self.trainEpochs(1)