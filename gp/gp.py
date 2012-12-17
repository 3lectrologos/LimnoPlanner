import numpy as np
import numpy.linalg


class GP(object):
    def __init__(self, kernel, d=2):
        self.kernel = kernel
        self.x = np.mat(np.zeros((0, d)))
        self.y = np.mat(np.zeros((0, 1)))
        self._update()

    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.x.shape[0]

    def _update(self):
        self.K = self.kernel(self.x)
        if self.K.shape[0] == 0:
            self.L = np.mat(np.zeros((0, 0)))
        else:
            # FIXME: Needs better handling (depending on noise)
            eps = np.diagflat(0.01*np.ones((1, len(self))))
            self.L = numpy.linalg.cholesky(self.K + eps)

    def add(self, x, y, update=True):
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == 1
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
