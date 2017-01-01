import snn.optimizers
import snn.optimizers.tests.test_functions as test_functions

import unittest
import numpy.testing as np_test


class TestRandomSearch(unittest.TestCase):

    def setUp(self):
        self.rs = snn.optimizers.get_method_class("random_search")

    def test_rs(self):
        for d in test_functions._functions:
            rs = self.rs()
            for i in range(0, 10):
                print("Random search: Point number %d" % i)
                start_point = test_functions._rnd_point(d["num"], d["bounds"])
                end_point = rs.start(d["f"], start_point)
                np_test.assert_allclose(end_point, d["min"], atol=d["atol"],
                                        err_msg="Start point was: {}".format(start_point))
