import unittest

import numpy as np
import numpy.testing as np_test

from core.Optimized_SNN import OptSNN
from network import SNN


class TestOptSNN(unittest.TestCase):

    def setUp(self):
        self.input = 3
        self.layer_count = 2
        self.neuron_count = 10
        self.fill_value = 1.0

        prev_layer_out = self.input
        self.snn = SNN.SNN(self.input)
        for i in range(0, self.layer_count):
            layer = SNN.Layer()
            for j in range(0, self.neuron_count):
                weights = np.zeros(prev_layer_out)
                weights.fill(self.fill_value)
                layer.add_neurons(SNN.Neuron("linear", weights))
            self.snn.add_layer(layer)
            prev_layer_out = 10
        layer = SNN.Layer()
        weights = np.zeros(self.snn.net_output_len())
        weights.fill(self.fill_value)
        layer.add_neurons(SNN.Neuron("linear", weights))
        self.snn.add_layer(layer)
        self.snn = OptSNN(self.snn)

    def test_base_creation(self):
        expected_weight_count = self.input * self.neuron_count +\
                                (self.layer_count - 1) * self.neuron_count ** 2 +\
                                self.neuron_count
        weights = self.snn.get_weights()
        self.assertEqual(weights.size, expected_weight_count)
        np_test.assert_array_equal(weights, self.fill_value)

    def test_snn_gives_right_out(self):
        input = np.zeros(self.input)
        input.fill(5.0)

        expected_out = 1500
        real_out = self.snn.input(input)

        np_test.assert_allclose(real_out, expected_out, rtol=1e-2)

    def test_snn_error_zero(self):
        input = np.zeros(self.input)
        input.fill(3.0)
        input = np.array([input])

        out = np.array([900])
        self.snn.set_test_data(input, out)

        np_test.assert_allclose(self.snn.error(), 0.0, rtol=1e-2)

    def test_weights_can_be_set(self):
        new_weights = np.zeros(self.snn.get_weights().size)
        new_weights.fill(10.0)
        self.snn.set_weights(new_weights)

        np_test.assert_array_equal(self.snn.get_weights(), new_weights)

    def test_error_on_bad_input(self):
        input = np.zeros(self.input * 34)

        self.assertRaises(ValueError, lambda: self.snn.input(input))

    def test_error_on_bad_new_weights(self):
        new_weights = np.zeros(self.snn.get_weights().size * 3)

        self.assertRaises(AssertionError, lambda: self.snn.set_weights(new_weights))

    def test_snn_gives_right_error(self):
        new_weights = np.zeros(self.snn.get_weights().size)
        new_weights.fill(2.0)
        self.snn.set_weights(new_weights)

        input = np.zeros(self.input)
        input.fill(1.0)
        input = np.array([input])

        out = np.array([300])

        self.snn.set_test_data(input, out)
        expected_error = 2100.0
        np_test.assert_allclose(self.snn.error(), expected_error, rtol=0.5)
