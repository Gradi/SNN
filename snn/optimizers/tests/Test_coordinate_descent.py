import snn.optimizers
import snn.optimizers.tests.test_functions as test_functions

import unittest
import numpy.testing as np_test


class TestCoordinateDescent(unittest.TestCase):

    def setUp(self):
        self.cd = snn.optimizers.get_method_class("coordinate_descent")

    def test_cd(self):
        for d in test_functions._functions:
            cd = self.cd(eps=1e-20, maxIter=20000, h=0.5)
            for i in range(0, 10):
                print("Coordinate descent: Point number %d" % i)
                start_point = test_functions._rnd_point(d["num"], d["bounds"])
                end_point = cd.start(d["f"], start_point)
                np_test.assert_allclose(end_point, d["min"], atol=d["atol"],
                                        err_msg="Start point was: {}".format(start_point))
