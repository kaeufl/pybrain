__author__ = 'Paul Kaeufl, kaeufl@geo.uu.nl'

import numpy as np
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, \
                              BiasUnit, FullConnection

class MixtureDensityNetwork(FeedForwardNetwork):
    """Mixture density network
    """
    def __init__(self, H, di, dy, M=1, **args):
        FeedForwardNetwork.__init__(self, **args)
        self.M = M
        self.c = dy
        dy = (2+self.c)*self.M
        self.addInputModule(LinearLayer(di, name = 'i'))
        self.addModule(TanhLayer(H, name = 'h'))
        self.addModule(BiasUnit('bias'))
        self.addOutputModule(LinearLayer(dy, name = 'o'))
        self.addConnection(FullConnection(self['i'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['h']))
        self.addConnection(FullConnection(self['bias'], self['o']))
        self.addConnection(FullConnection(self['h'], self['o']))
        self.sortModules()

    def convertToFastNetwork(self):
        cnet = FeedForwardNetwork.convertToFastNetwork(self)
        cnet.M = self.M
        cnet.c = self.c
        return cnet