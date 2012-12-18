import numpy.matlib as np
import scipy.linalg
import numpy.linalg


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
            self.L = np.zeros((0, 0))
        else:
            sn2 = np.exp(2*self.kernel.lik)
            self.K = self.K + sn2*np.eye(len(self))
            self.L = scipy.linalg.cholesky(self.K, lower=True)

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

    def clear(self):
        self.remove(len(self))

    def inf(self, x, meanonly=False):
        x = np.asmatrix(x)
        assert x.shape[1] == self.d
        n = x.shape[0]
        # Handle empty test set
        if n == 0:
            return (np.zeros((0, 1)), np.zeros((0, 1)))
        ms = self.kernel.mean*np.ones((n, 1))
        Kbb = self.kernel(x, diag=True)
        # Handle empty training set
        if len(self) == 0:
            return (ms, np.asmatrix(np.diag(Kbb)).T)
        Kba = self.kernel(x, self.x)
        m = self.kernel.mean*np.ones((len(self), 1))
        fm = ms + Kba*scipy.linalg.cho_solve((self.L, True), self.y - m,
                                             overwrite_b=True)
        if meanonly:
            return fm
        else:
            W = scipy.linalg.cho_solve((self.L, True), Kba.T)
            fv = np.asmatrix(Kbb - np.sum(np.multiply(Kba.T, W), axis=0).T)
            # W = np.asmatrix(scipy.linalg.solve(self.L, Kba.T, lower=True))
            # fv = np.asmatrix(Kbb - np.sum(np.power(W, 2), axis=0).T)
            return (fm, fv)

    def minfo(self, x):
        x = np.asmatrix(x)
        assert x.shape[1] == self.d
        n = x.shape[0]
        # Handle empty test set
        if n == 0:
            return 0
        Kbb = self.kernel(x)
        # Handle empty training set
        if len(self) == 0:
            S = Kbb
        else:
            Kba = self.kernel(x, self.x)
            S = Kbb - Kba*scipy.linalg.cho_solve((self.L, True), Kba.T)
        sn2 = np.exp(2*self.kernel.lik)
        (sng, mi) = numpy.linalg.slogdet(np.eye(n) + S/sn2)
        return 0.5*mi
