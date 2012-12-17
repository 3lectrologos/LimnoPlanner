import unittest
import numpy.matlib as np
import plan.gp.gp
import plan.gp.kernels


def almostEqual(A, B, eps=0.01):
    return np.amax(abs(A - B)) < eps

class GPBasicsTest(unittest.TestCase):
    def setUp(self):
        hyp = {'mean': 0, 'cov': [-1.5, -1.3, 5], 'lik': -1}
        k = plan.gp.kernels.SE(hyp)
        self.gp = plan.gp.gp.GP(k)

    def test_len(self):
        self.assertEqual(len(self.gp), 0)

    def test_add(self):
        self.gp.add(np.ones((5, 2)), np.ones((5, 1)))
        self.assertEqual(len(self.gp), 5)

    def test_remove(self):
        self.gp.add(np.ones((5, 2)), np.ones((5, 1)))
        self.gp.remove(3)
        self.assertEqual(len(self.gp), 2)


class GPInfTest(unittest.TestCase):
    def setUp(self):
        hyp = {'mean': 1, 'cov': [-1.5, -1.3, 1], 'lik': -1}
        k = plan.gp.kernels.SE(hyp)
        self.gp = plan.gp.gp.GP(k)
        x = np.mat([[0, 0], [0, 0.5], [1, 0]])
        y = np.mat([2, 1, 3]).T
        self.gp.add(x, y)
        self.z = np.mat([[0.5, 0.5], [0.25, 0.5]])

    def test_inf_mean(self):
        (fm, _) = self.gp.inf(self.z)
        fmtrue = np.mat([1.0299129, 1.0030889]).T
        self.assertTrue(almostEqual(fmtrue, fm, 0.00001))

    def test_multi_inf_mean(self):
        (fm1, _) = self.gp.inf(self.z)
        (fm2, _) = self.gp.inf(self.z)
        self.assertTrue(almostEqual(fm1, fm2, 0.00001))
