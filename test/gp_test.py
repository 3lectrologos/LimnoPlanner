import unittest
import numpy as np
import plan.gp.gp


class GPTest(unittest.TestCase):
    def test_len(self):
        tmp = plan.gp.gp.GP(2)
        self.assertEqual(len(tmp), 0)

    def test_add(self):
        tmp = plan.gp.gp.GP(2)
        tmp.add(np.zeros((5, 2)), np.zeros((5, 1)))
        self.assertEqual(len(tmp), 5)

    def test_remove(self):
        tmp = plan.gp.gp.GP(2)
        tmp.add(np.zeros((5, 2)), np.zeros((5, 1)))
        tmp.remove(3)
        self.assertEqual(len(tmp), 2)


if __name__ == '__main__':
    unittest.main()
