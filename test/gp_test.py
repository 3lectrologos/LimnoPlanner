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

    def test_clear(self):
        self.gp.add(np.ones((5, 2)), np.ones((5, 1)))
        self.gp.clear()
        self.assertEqual(0, len(self.gp))

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

    def test_multiple_inf_mean(self):
        (fm1, _) = self.gp.inf(self.z)
        (fm2, _) = self.gp.inf(self.z)
        self.assertTrue(almostEqual(fm1, fm2, 0.00001))

    def test_inf_var(self):
        (_, fv) = self.gp.inf(self.z)
        fvtrue = np.mat([7.339546, 5.321190]).T
        self.assertTrue(almostEqual(fvtrue, fv, 0.00001))

    def test_multiple_inf_var(self):
        (_, fv1) = self.gp.inf(self.z)
        (_, fv2) = self.gp.inf(self.z)
        self.assertTrue(almostEqual(fv1, fv2))

    def test_inf_empty(self):
        (fm, fv) = self.gp.inf(np.zeros((0, 2)))
        self.assertEqual((0, 1), fm.shape)
        self.assertEqual((0, 1), fv.shape)

    def test_inf_notrain(self):
        empty = self.gp.clear()
        (fm, fv) = self.gp.inf(self.z)
        fmtrue = np.mat([1, 1]).T
        fvtrue = np.mat([7.389056, 7.389056]).T
        self.assertTrue(almostEqual(fmtrue, fm, 0.00001))
        self.assertTrue(almostEqual(fvtrue, fv, 0.00001))
