import numpy.matlib as np
import scipy.spatial.distance as ssd

class Kernel(object):
    def __init__(self, hyp):
        self.mean = hyp['mean']
        self.cov = np.exp(hyp['cov'][:-1])
        self.covd = np.asmatrix(np.diagflat(1/self.cov))
        self.sf2 = np.exp(2*hyp['cov'][-1])
        self.lik = hyp['lik']

class SE(Kernel):
    def __call__(self, x, z=[], diag=False):
        if x.shape[0] == 0:
            return np.zeros((0, 0))
        if diag:
            return self.sf2*np.ones((x.shape[0], 1))
        sx = x*self.covd
        if z == []:
            K = np.asmatrix(ssd.squareform(ssd.pdist(sx, 'sqeuclidean')))
        else:
            sz = z*self.covd
            K = np.asmatrix(ssd.cdist(sx, sz, 'sqeuclidean'))
        K = self.sf2*np.exp(-K/2)
        return K
