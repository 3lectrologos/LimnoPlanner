import numpy as np


class GP(object):
    def __init__(self, kernel, d=2):
        self.kernel = kernel
        self.x = np.zeros((0, d))
        self.y = np.zeros((0, 1))

    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.x.shape[0]

    def add(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == 1
        if x.shape[0] > 0:
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))

    def remove(self, n):
        assert n >= 0
        if n > 0:
            self.x = np.delete(self.x, np.s_[-n:], 0)
            self.y = np.delete(self.y, np.s_[-n:], 0)
