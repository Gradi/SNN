import snn.optimizers
import snn.optimizers.tests.test_functions as test_functions

import unittest
import numpy.testing as np_test


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        self.gd = snn.optimizers.get_method_class("gradient_descent")

    def test_gd(self):
        for d in test_functions._functions:
            self.bounds = d["bounds"]
            gradient_descent = self.gd(maxIter=10000)
            for i in range(0, 10):
                print("Gradient descent: Point number %d" % i)
                start_point = test_functions._rnd_point(d["num"], self.bounds)
                end_point = gradient_descent.start(d["f"], start_point)
                np_test.assert_allclose(end_point, d["min"], atol=d["atol"],
                                        err_msg="Start point was %s" % str(start_point))