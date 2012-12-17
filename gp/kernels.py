import numpy as np
import scipy.spatial.distance as ssd

class Kernel(object):
    def __init__(self, hyp):
        self.mean = hyp['mean']
        self.cov = np.mat(np.exp(hyp['cov'][:-1]))
        self.covd = np.diagflat(1/self.cov)
        self.sf2 = np.exp(2*hyp['cov'][-1])
        self.lik = hyp['lik']

class SE(Kernel):
    def __call__(self, x, z=[]):
        sx = x*self.covd
        if z == []:
            K = ssd.squareform(ssd.pdist(sx, 'sqeuclidean'))
        else:
            sz = z*self.covd
            K = ssd.cdist(sx, sz, 'sqeuclidean')
        K = np.mat(self.sf2*np.exp(-K/2))
        return K
