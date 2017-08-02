#!/usr/bin/python
from __future__ import division
import os
import sys
import math
import numpy as np
import scipy
import scipy.special
import argparse

# inverse digamma/psi function
def invpsi(x):
    sign = lambda x: x and (1, -1)[x<0]
    l=1
    y=math.exp(x)
    while l>10e-8:
        y=y+l*sign(x-scipy.special.psi(y))
        l=l/2
    return y

class GammaMLE():
# alpha indicates shape parameter
# beta indicates rate parameter
    def __init__(self, a=1.0, b=1.0):
        self.alpha=a
        self.beta=b

    def mle(self, data, tot=1e-8, MAX_ITER=500):
        log_mean=sum([math.log(d) for d in data])/len(data)

        mean_log=math.log(sum(data)/len(data))

        error=1
        i=0
        while error > tot and i<MAX_ITER:
            a=invpsi(math.log(self.alpha)-mean_log+log_mean)

            # log-likelihood / n
            l=a*math.log(a)-a*mean_log-math.log(scipy.special.gamma(a))+(a-1)*log_mean-a

            # lower bound of l
            cost=a*math.log(self.alpha)-self.alpha-a*mean_log-math.log(scipy.special.gamma(a))+(a-1)*log_mean

            error=(self.alpha-a)*(self.alpha-a)

            self.alpha=a

            i+=1

            print '[Estimation] Iteration: %s, error: %s, alpha: %s, cost: %s, l: %s' % (i, error, a, cost, l)

        self.beta=self.alpha/(sum(data)/len(data))

        print '[MLE] alpha=%s, beta=%s' % (self.alpha, self.beta)

def parse_arguments():
    parser = argparse.ArgumentParser(description="MLE of gamma distribution.")
    
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha (shape parameter)')

    parser.add_argument('--beta', type=float, default=1.0, help='beta (srate parameter)')
    
    parser.add_argument('--n', type=int, default=100000, help='number of samples for MLE')

    return parser.parse_args()

args_parser=parse_arguments()

def run():
    gamma=GammaMLE()
    
    data=np.random.gamma(args_parser.alpha, 1.0/args_parser.beta, args_parser.n)

    gamma.mle(data)

if __name__=='__main__':
    run()
