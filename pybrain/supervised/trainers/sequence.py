'''
Created on Oct 5, 2011

@author: kaeufl
'''
from scipy import dot, zeros

from rprop import RPropMinusTrainer

class SequenceTrainer(RPropMinusTrainer):
    """SequenceTrainer that uses only the last target value of each sequence for error 
    calculation."""
    
    def __init__(self, module, etaminus=0.5, etaplus=1.2, deltamin=1.0e-6, deltamax=5.0, delta0=0.1, **kwargs):
        self.sequential_target = kwargs['sequential_target']
        del(kwargs['sequential_target'])
        RPropMinusTrainer.__init__(self, module, etaminus, etaplus, deltamin, deltamax, delta0, **kwargs)
        
    
    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield
        the gradient."""
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            
            # if sequenital_target is set, use error due to target value only for the last sample of the sequence
            if offset == 1 or self.sequential_target:
                outerr = target - self.module.outputbuffer[offset]
            else:
                outerr = zeros(target.shape)
            
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)

        return error, ponderation
