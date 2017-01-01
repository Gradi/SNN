import snn.optimizers
import snn.optimizers.tests.test_functions as test_functions
from snn.optimizers.optimizer_manager import OptManager

import unittest
import numpy.testing as np_test


class TestAllOptimizers(unittest.TestCase):

    def test_all(self):
        opt_manager = OptManager(eps=1e-5)

        for optimizer in opt_manager:
            for func in test_functions._functions:
                for i in range(0, 10):
                    start_point = test_functions._rnd_point(func["num"],
                                                            func["bounds"])
                    msg = "Optimizer name: {}, start point was: {}"\
                          .format(type(optimizer).__name__, start_point)
                    end_point = optimizer.start(func["f"], start_point)
                    np_test.assert_allclose(end_point, func["min"],
                                            atol=func["atol"],
                                            err_msg=msg)
