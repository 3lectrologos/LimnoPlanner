import unittest
import numpy as np
import plan.gp.gp
import plan.gp.kernels


class GPTest(unittest.TestCase):
    def setUp(self):
        hyp = {'mean': 0, 'cov': [-1.5, -1.3, 5], 'lik': -1}
        k = plan.gp.kernels.SE(hyp)
        self.tmp = plan.gp.gp.GP(k)

    def test_len(self):
        self.assertEqual(len(self.tmp), 0)

    def test_add(self):
        self.tmp.add(np.ones((5, 2)), np.ones((5, 1)))
        self.assertEqual(len(self.tmp), 5)

    def test_remove(self):
        self.tmp.add(np.ones((5, 2)), np.ones((5, 1)))
        self.tmp.remove(3)
        self.assertEqual(len(self.tmp), 2)


if __name__ == '__main__':
    unittest.main()
