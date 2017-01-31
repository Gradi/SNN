import unittest
import numpy.testing as np_tests
import numpy as np
from snn.optimizers.utils import gradient as gradient


class TestGradient(unittest.TestCase):

    def f(self, x):
        return np.sum(x) + np.sum(np.square(x)) + np.sum(np.power(x, 3))

    def dfx(self, x):
        x, y, z = x
        return 3 * np.square(x) + 2 * x + 1.0

    def dfy(self, x):
        x, y, z = x
        return 3 * np.square(y) + 2 * y + 1.0

    def dfz(self, x):
        x, y, z = x
        return 3 * np.square(z) + 2 * z + 1.0

    def test_gradient(self):
        x = -10 + np.random.rand(5000, 3) * 20
        good_gr = list()
        for point in x:
            dx = self.dfx(point)
            dy = self.dfy(point)
            dz = self.dfz(point)
            good_gr.append([dx, dy, dz])
        good_gr = np.array(good_gr)
        num_gr = gradient(self.f, x, 1e-5, norm=False)
        self.assertEqual(num_gr.shape, good_gr.shape, msg="Shapes")
        self.assertEqual(num_gr.size, good_gr.size, msg="Sizes")
        np_tests.assert_allclose(num_gr, good_gr)


