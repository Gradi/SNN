import optimizers
import optimizers.tests.test_functions as test_functions

import unittest
import numpy.testing as np_test


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        self.gd = optimizers.get_method_class("gradient_descent")

    def test_gd(self):
        for d in test_functions._functions:
            self.bounds = d["bounds"]
            gradient_descent = self.gd({"maxIter": 10000})
            for i in range(0, 10):
                print("Gradient descent: Point number %d" % i)
                start_point = test_functions._rnd_point(d["num"], self.bounds)
                end_point = gradient_descent.start(d["f"], start_point, self._check_bounds)
                np_test.assert_allclose(end_point, d["min"], atol=d["atol"])

    def _check_bounds(self, x):
        test_functions._check_bounds(x, self.bounds)