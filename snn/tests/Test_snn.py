import unittest

import numpy as np
import numpy.testing as np_test

from snn.network.snn import SNN, load_from_json, Neuron, Layer


class TestSNN(unittest.TestCase):

    def test_base_creation(self):
        net = SNN(3)
        neurons = list()
        for i in range(0, 5):
            neurons.append(Neuron("linear", np.array([1.0, 1.0, 1.0])))
        l = Layer()
        l.add_neurons(neurons)
        net.add_layer(l)
        l = Layer()
        neurons = list()
        for i in range(0, 5):
            neurons.append(Neuron("linear", np.array([1.0, 1.0, 1.0, 1.0, 1.0])))
        l.add_neurons(neurons)
        net.add_layer(l)

        net_weights = net.get_weights()

        self.assertEqual(len(net_weights), 40)
        self.assertEqual(np.equal(net_weights, 1.0).all(), True)

        self.assertEqual(net.net_output_len(), 5)

    def test_right_output(self):
        net = SNN(2)
        neurons = list()
        for i in range(0, 5):
            neurons.append(Neuron("linear", np.array([1.0, 1.0])))
        l = Layer()
        l.add_neurons(neurons)
        net.add_layer(l)
        l = Layer()
        neurons = list()
        for i in range(0, 5):
            neurons.append(Neuron("linear", np.array([1.0, 1.0, 1.0, 1.0, 1.0])))
        l.add_neurons(neurons)
        net.add_layer(l)
        l = Layer()
        l.add_neurons(Neuron("linear", np.array([1.0, 1.0, 1.0, 1.0, 1.0])))
        net.add_layer(l)

        expected_output = 100.0
        real_output = net.input(np.array([2.0, 2.0]))

        np_test.assert_allclose(real_output, expected_output)

    def test_net_fails_on_bad_input_001(self):
        net = SNN(43)
        neurons = list()
        for i in range(0, 10):
            neurons.append(Neuron("linear", np.random.rand(43)))
        l = Layer()
        l.add_neurons(neurons)
        net.add_layer(l)

        input = np.random.rand(42)

        self.assertRaises(ValueError, lambda: net.input(input))

    def test_net_fails_on_bad_input_002(self):
        net = SNN(43)
        neurons = list()
        for i in range(0, 10):
            neurons.append(Neuron("linear", np.random.rand(43)))
        l = Layer()
        l.add_neurons(neurons)
        net.add_layer(l)

        input = np.random.rand(44)

        self.assertRaises(ValueError, lambda: net.input(input))

    def test_net_fails_on_bad_input_003(self):
        net = SNN(43)
        neurons = list()
        for i in range(0, 10):
            neurons.append(Neuron("linear", np.random.rand(43)))
        l = Layer()
        l.add_neurons(neurons)
        net.add_layer(l)

        input = np.array([])

        self.assertRaises(ValueError, lambda: net.input(input))

    def test_error(self):
        net = SNN(3)
        l = Layer()
        for i in range(0, 5):
            l.add_neurons(Neuron("linear", np.array([2.0, 2.0, 2.0])))
        net.add_layer(l)
        l = Layer()
        for i in range(0, 5):
            l.add_neurons(Neuron("linear", np.array([2.0, 2.0, 2.0, 2.0, 2.0])))
        net.add_layer(l)
        l = Layer()
        l.add_neurons(Neuron("linear", np.array([2.0, 2.0, 2.0, 2.0, 2.0])))
        net.add_layer(l)

        input = np.array([[5.0, 5.0, 5.0]])
        test_result = np.array([375.0])
        expected_error = 2625
        net.set_test_inputs(input, test_result)
        real_error = net.error()

        np_test.assert_allclose(real_error, expected_error)

    def test_save_to_json_noweights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([1.0, 1.0, 1.0])
        for i in range(0, 50):
            neuron = Neuron("linear", weights)
            layer.add_neurons(neuron)
        net.add_layer(layer)

        json_str = net.to_json(with_weights=False)
        restored_net = load_from_json(json_str)

        self.assertEqual(len(restored_net.layers()), 1)
        layer = restored_net.layers()[0]
        self.assertEqual(layer.in_len(), 3)
        self.assertEqual(layer.out_len(), 50)
        for neuron in layer:
            self.assertEqual(neuron.func_name(), "linear")
            self.assertEqual(neuron.w_len(), 3)
            self.assertEqual(neuron.f_len(), 0)

    def test_save_to_json_weights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([1.0, 1.0, 1.0])
        for i in range(0, 50):
            neuron = Neuron("linear", weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        json_str = net.to_json(with_weights=True)
        restored_net = load_from_json(json_str)

        self.assertEqual(len(restored_net.layers()), 1)
        layer = restored_net.layers()[0]
        self.assertEqual(layer.in_len(), 3)
        self.assertEqual(layer.out_len(), 50)
        for neuron in layer:
            self.assertEqual(neuron.func_name(), "linear")
            self.assertEqual(neuron.w_len(), 3)
            self.assertEqual(neuron.f_len(), 0)
            np_test.assert_array_equal(neuron.get_input_weights(), weights)

    def test_returns_only_input_weights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([3.14, 3.14, 3.14])
        for i in range(0, 50):
            neuron = Neuron("linear", weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        input_weights = net.get_weights("input")
        np_test.assert_allclose(input_weights, 3.14)
        func_weights = net.get_weights("func")
        self.assertEqual(func_weights.size, 0)

    def test_returns_only_func_weights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([3.14, 3.14, 3.14])
        func_weights = np.array([2.5])
        for i in range(0, 50):
            neuron = Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        func_weights = net.get_weights("func")
        self.assertEqual(func_weights.size, 50)
        np_test.assert_allclose(func_weights, 2.5)

    def test_set_only_input_weights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([3.14, 3.14, 3.14])
        func_weights = np.array([2.5])
        for i in range(0, 50):
            neuron = Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        input_weights = net.get_weights("input")
        input_weights.fill(0)
        net.set_weights(input_weights, "input")
        input_weights = net.get_weights("input")
        np_test.assert_allclose(input_weights, 0)

    def test_set_only_func_weights(self):
        net = SNN(3)
        layer = Layer()
        weights = np.array([3.14, 3.14, 3.14])
        func_weights = np.array([2.5])
        for i in range(0, 50):
            neuron = Neuron("linear", weights.copy(), func_weights.copy())
            layer.add_neurons(neuron)
        net.add_layer(layer)

        func_weights = net.get_weights("func")
        func_weights.fill(0)
        net.set_weights(func_weights, "func")
        func_weights = net.get_weights("func")
        np_test.assert_allclose(func_weights, 0)
