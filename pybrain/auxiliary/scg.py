"""
 Scaled Conjugate Gradient algorithm from
  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
  by Martin F. Moller
  Neural Networks, vol. 6, pp. 525-533, 1993

  Adapted for use with pybrain.supervised.scg.SCGTrainer from the adaption by 
  Chuck Anderson of the Matlab implementation by Nabney as part of the netlab 
  library.
"""
from copy import copy
import numpy as np
import sys
from math import sqrt, ceil

class SCG():
    def __init__(self, x0, f, df, trainer, nIterations = 1000, 
                 xPrecision = None, fPrecision = None, evalFunc = None):
        """
        Initialize SCG with initial guess 'x0', objective function 'f(x, trainer)' and
        derivative 'df(x, trainer)' and a reference to the trainer object keeping the SCG
        instance.
        """
        self.f = lambda x: f(x, self.trainer)
        self.gradf = lambda x: df(x, self.trainer)
        self.trainer = trainer
        
        self.nIterations = nIterations
        self.xPrecision = xPrecision or 0.001 * np.mean(x0)
        self.fPrecision = fPrecision or 0.001 * np.mean(self.f(x0))
        self.xtracep = True
        self.ftracep = True
        self.sigma0 = 1.0e-4
        self.nvars = len(x0)
        self.evalFunc = evalFunc or (lambda x: "Eval "+str(x))
        
        self.success = True        # Force calculation of directional derivs.
        self.nsuccess = 0        # nsuccess counts number of successes.
        self.beta = 1.0        # Initial scale parameter.
        self.betamin = 1.0e-15       # Lower bound on scale.
        self.betamax = 1.0e100     # Upper bound on scale.
        self.floatPrecision = sys.float_info.epsilon
        self.j = 1       # j counts number of iterations.
        
        # initialize a few things
        self.fold = self.f(x0)
        self.fnow = self.fold
        self.gradnew = self.gradf(x0)
        self.gradold = copy(self.gradnew)
        self.d = -self.gradnew        # Initial search direction.
        
        if self.xtracep:
            self.xtrace = np.zeros((self.nIterations+1,len(x0)))
            self.xtrace[0,:] = x0
        else:
            self.xtrace = None
        if self.ftracep:
            self.ftrace = np.zeros(nIterations+1)
            self.ftrace[0] = self.fold
        else:
            self.ftrace = None
    
    def scg(self, x):
        """perform one iteration"""
        ### Main optimization loop.
        #while j <= nIterations:
    
        # Calculate first and second directional derivatives.
        if self.success:
            self.mu = np.dot(self.d, self.gradnew)
            if self.mu==np.nan: print "mu is NaN"
            if self.mu >= 0:
                self.d = -self.gradnew
                self.mu = np.dot(self.d, self.gradnew)
            self.kappa = np.dot(self.d, self.d)
            if self.kappa < self.floatPrecision:
                return {'x':x, 'f':self.fnow, 'nIterations':self.j, 'xtrace':self.xtrace[:self.j,:], 'ftrace':self.ftrace[:self.j],
                        'reason':"limit on machine precision"}
            sigma = self.sigma0/sqrt(self.kappa)
            xplus = x + sigma * self.d
            gplus = self.gradf(xplus)
            self.theta = np.dot(self.d, gplus - self.gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = self.theta + self.beta * self.kappa
        if delta is np.nan: print "delta is NaN"
        if delta <= 0:
            delta = self.beta * self.kappa
            self.beta = self.beta - self.theta/self.kappa
        alpha = -self.mu/delta
        
        ## Calculate the comparison ratio.
        xnew = x + alpha * self.d
        fnew = self.f(xnew)
        Delta = 2 * (fnew - self.fold) / (alpha*self.mu)
        if Delta is not np.nan and Delta  >= 0:
            self.success = True
            self.nsuccess += 1
            x = xnew
            self.fnow = fnew
        else:
            self.success = False
            self.fnow = self.fold
        if self.xtracep:
            self.xtrace[self.j,:] = x
        if self.ftracep:
            self.ftrace[self.j] = fnew

        #if self.j % ceil(self.nIterations/10) == 0:
        #    print "SCG: Iteration",self.j,"fValue",self.evalFunc(self.fnow),"Scale",self.beta

        if self.success:
        ## Test for termination

        ##print(c(max(abs(alpha*d)),max(abs(fnew-fold))))
      
            if max(abs(alpha*self.d)) < self.xPrecision:
                return {'x':x, 'f':self.fnow, 'nIterations':self.j, 'xtrace':self.xtrace[:self.j,:], 'ftrace':self.ftrace[:self.j],
                        'reason':"limit on x Precision"}
            elif abs(fnew-self.fold) < self.fPrecision:
                return {'x':x, 'f':self.fnow, 'nIterations':self.j, 'xtrace':self.xtrace[:self.j,:], 'ftrace':self.ftrace[:self.j],
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                self.fold = fnew
                self.gradold = self.gradnew
                self.gradnew = self.gradf(x)
                #print "gradold",gradold
                #print "gradnew",gradnew
                ## If the gradient is zero then we are done.
                if np.dot(self.gradnew, self.gradnew) == 0:
                    return {'x':x, 'f':self.fnow, 'nIterations':self.j, 'xtrace':self.xtrace[:self.j,:], 'ftrace':self.ftrace[:self.j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if Delta is np.nan or Delta < 0.25:
            self.beta = min(4.0*self.beta, self.betamax)
        elif Delta > 0.75:
            self.beta = max(0.5*self.beta, self.betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start 
        ## in direction of negative gradient after nparams steps.
        if self.nsuccess == self.nvars:
            self.d = -self.gradnew
            self.nsuccess = 0
        elif self.success:
            self.gamma = np.dot(self.gradold - self.gradnew, self.gradnew/self.mu)
            #print "gamma",gamma
            self.d = self.gamma * self.d - self.gradnew
            #print "end d",d
        self.j += 1

        ## If we get here, then we haven't terminated in the given number of 
        ## iterations.

        ##print("Did not converge.")
        return {'x':x, 'f':self.fnow, 'nIterations':self.j, 
                'xtrace':self.xtrace[:self.j,:], 'ftrace':self.ftrace[:self.j],
                'reason':"did not converge"}
