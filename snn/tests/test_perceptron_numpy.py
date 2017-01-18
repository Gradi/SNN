import unittest, logging

import numpy as _np
import numpy.testing as _np_tests

from snn.network.perceptron_numpy import PerceptronNumpy as _PN
import snn


class TestPerceptronNumpy(unittest.TestCase):

    def setUp(self):
        self.input_count = 10
        self.net = _PN(self.input_count)
        self.net.add_layer(snn.make_layer("linear", 10))
        self.net.add_layer(snn.make_layer("quadratic", 20))
        self.net.add_layer(snn.make_layer("linear", 1))

    def test_base_creation(self):
        net_weights = self.net.get_weights()
        weights_count = 10 * 10 + 20 * 10 + 1 * 20
        self.assertEqual(len(net_weights), weights_count)
        self.assertEqual(self.net.net_output_len(), 1)

    def test_right_output(self):
        net_weights = self.net.get_weights()
        net_weights.fill(1.0)
        self.net.set_weights(net_weights)

        input = _np.empty(self.input_count, dtype=_np.float)
        input.fill(5.0)

        expected_output = 5000000
        real_output = self.net.input(input)

        _np_tests.assert_equal(real_output, expected_output)

    def test_net_fails_on_bad_input_001(self):
        input = _np.random.rand(42)

        self.assertRaises(ValueError, lambda: self.net.input(input))

    def test_net_fails_on_bad_input_002(self):
        input = _np.random.rand(44)

        self.assertRaises(ValueError, lambda: self.net.input(input))

    def test_net_fails_on_bad_input_003(self):
        input = _np.array([])

        self.assertRaises(ValueError, lambda: self.net.input(input))

    def test_error(self):
        net_weights = self.net.get_weights()
        net_weights.fill(2.0)
        self.net.set_weights(net_weights)
        input = _np.empty(self.input_count, dtype=_np.float)
        input.fill(5.0)
        input = _np.array([input])
        test_result = _np.array([5000000.0])
        expected_error = 155000000.08391486
        self.net.set_test_inputs(input, test_result)
        real_error = self.net.error()

        _np_tests.assert_allclose(real_error, expected_error)

    def test_save_to_json_noweights(self):

        json_str = self.net.to_json(with_weights=False)
        restored_net = _PN.load_from_json(json_str)

        self.assertEqual(len(restored_net.layers()), 3)
        layer = restored_net.layers()[0]
        self.assertEqual(layer.in_len(), 10)
        self.assertEqual(layer.out_len(), 10)
        layer = restored_net.layers()[1]
        self.assertEqual(layer.in_len(), 10)
        self.assertEqual(layer.out_len(), 20)
        layer = restored_net.layers()[2]
        self.assertEqual(layer.in_len(), 20)
        self.assertEqual(layer.out_len(), 1)

    def test_save_to_json_weights(self):
        net_weights = self.net.get_weights()
        net_weights.fill(1.0)
        self.net.set_weights(net_weights)
        json_str = self.net.to_json(with_weights=True)
        restored_net = _PN.load_from_json(json_str)

        self.assertEqual(len(restored_net.layers()), 3)
        layer = restored_net.layers()[0]
        self.assertEqual(layer.in_len(), 10)
        self.assertEqual(layer.out_len(), 10)
        layer = restored_net.layers()[1]
        self.assertEqual(layer.in_len(), 10)
        self.assertEqual(layer.out_len(), 20)
        layer = restored_net.layers()[2]
        self.assertEqual(layer.in_len(), 20)
        self.assertEqual(layer.out_len(), 1)
        self.assertEqual(_np.equal(restored_net.get_weights(), 1.0).all(), True)

    def test_returns_only_input_weights(self):
        net_weights = self.net.get_weights()
        net_weights.fill(1.0)
        self.net.set_weights(net_weights)
        input_weights = self.net.get_weights("input")
        _np_tests.assert_equal(input_weights, 1.0)
        func_weights = self.net.get_weights("func")
        self.assertEqual(func_weights.size, 0)

    def test_returns_only_func_weights(self):
        net = _PN(3)
        layer = snn.Layer()
        weights = _np.array([3.14, 3.14, 3.14])
        func_weights = _np.array([2.5])
        for i in range(0, 50):
            neuron = snn.Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        func_weights = net.get_weights("func")
        self.assertEqual(func_weights.size, 50)
        self.assertEqual(_np.equal(func_weights, 2.5).all(), True)

    def test_set_only_input_weights(self):
        net = _PN(3)
        layer = snn.Layer()
        weights = _np.array([3.14, 3.14, 3.14])
        func_weights = _np.array([2.5])
        for i in range(0, 50):
            neuron = snn.Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        input_weights = net.get_weights("input")
        input_weights.fill(0)
        net.set_weights(input_weights, "input")
        input_weights = net.get_weights("input")
        _np_tests.assert_equal(input_weights, 0)

    def test_set_only_func_weights(self):
        net = _PN(3)
        layer = snn.Layer()
        weights = _np.array([3.14, 3.14, 3.14])
        func_weights = _np.array([2.5])
        for i in range(0, 50):
            neuron = snn.Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        func_weights = net.get_weights("func")
        func_weights.fill(0)
        net.set_weights(func_weights, "func")
        func_weights = net.get_weights("func")
        _np_tests.assert_equal(func_weights, 0)
