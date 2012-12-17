import numpy.matlib as np
import scipy.linalg


class GP(object):
    def __init__(self, kernel, d=2):
        self.d = d
        self.kernel = kernel
        self.x = np.zeros((0, d))
        self.y = np.zeros((0, 1))
        self._update()

    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.x.shape[0]

    def _update(self):
        self.K = self.kernel(self.x)
        if self.K.shape[0] == 0:
            self.cho = np.zeros((0, 0))
        else:
            sn2 = np.exp(2*self.kernel.lik)
            self.K = self.K + sn2*np.eye(len(self))
            self.cho = scipy.linalg.cho_factor(self.K)

    def inf(self, x):
        assert x.shape[1] == self.d
        n = x.shape[0]
        x = np.asmatrix(x)
        # TODO: Handle x.shape[0] == 0 separately?
        Kba = self.kernel(x, self.x)
        m = self.kernel.mean*np.ones((len(self), 1))
        ms = self.kernel.mean*np.ones((n, 1))
        fm = ms + Kba*scipy.linalg.cho_solve(self.cho, self.y - m)
        
        return (fm, 0)

    def add(self, x, y, update=True):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.d
        assert y.shape[1] == 1
        x = np.asmatrix(x)
        y = np.asmatrix(y)
        if x.shape[0] > 0:
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))
        if update:
            self._update()

    def remove(self, n, update=True):
        assert n >= 0
        if n > 0:
            self.x = np.delete(self.x, np.s_[-n:], 0)
            self.y = np.delete(self.y, np.s_[-n:], 0)
        if update:
            self._update()
