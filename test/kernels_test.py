import unittest
import numpy.matlib as np
import numpy.testing
import plan.gp.kernels


def almostEqual(A, B, eps=0.01):
    return np.amax(abs(A - B)) < eps

class SETest(unittest.TestCase):
    def setUp(self):
        self.hyp = {'mean': 0, 'cov': [-1.5, -1.3, 5], 'lik': -1}
        self.k = plan.gp.kernels.SE(self.hyp)
        self.x = np.mat([[0, 0], [0, 0.5], [1, 0]])
        self.z = np.mat([[0.5, 0.5], [0.25, 0.5]])
    
    def test_SE_same(self):
        K = self.k(self.x)
        Ktrue = np.mat([[22026.46, 4093, 0.958],
                        [4093, 22026.46, 0.178],
                        [0.958, 0.178, 22026.46]])
        self.assertTrue(almostEqual(Ktrue, K))

    def test_SE_diff(self):
        K = self.k(self.x, self.z)
        Ktrue = np.mat([[332.40, 2184.98],
                        [1788.81, 11758.44],
                        [332.40, 14.41]])
        self.assertTrue(almostEqual(Ktrue, K))

    def test_SE_diag(self):
        K = self.k(self.x, diag=True)
        sf2 = np.exp(2*self.hyp['cov'][-1])
        Ktrue = sf2*np.ones((3, 1)).T
        self.assertTrue(almostEqual(Ktrue, K))
