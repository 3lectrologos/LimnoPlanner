import unittest
import numpy as np
import numpy.testing
import plan.gp.kernels


class KernelsTest(unittest.TestCase):
    def test_SE_same(self):
        hyp = {'mean': 0, 'cov': [-1.5, -1.3, 5], 'lik': -1}
        k = plan.gp.kernels.SE(hyp)
        x = np.mat([[0, 0], [0, 0.5], [1, 0]])
        K = k(x)
        Ktrue = np.mat([[22026.46, 4093, 0.958],
                        [4093, 22026.46, 0.178],
                        [0.958, 0.178, 22026.46]])
        self.assertTrue(np.amax(abs(K - Ktrue)) < 0.01)

    def test_SE_diff(self):
        hyp = {'mean': 0, 'cov': [-1.5, -1.3, 5], 'lik': -1}
        k = plan.gp.kernels.SE(hyp)
        x = np.mat([[0, 0], [0, 0.5], [1, 0]])
        z = np.mat([[0.5, 0.5], [0.25, 0.5]])
        K = k(x, z)
        Ktrue = np.mat([[332.40, 2184.98],
                        [1788.81, 11758.44],
                        [332.40, 14.41]])
        self.assertTrue(np.amax(abs(K - Ktrue)) < 0.01)
