import unittest

import numpy as _np
import numpy.testing as _np_tests

from snn.core import snn_core as snn


class TestNeuron(unittest.TestCase):

    def test_base_creation(self):
        w_len = 10
        f_len = 1
        weights = _np.linspace(1, 10, w_len)
        func_weights = _np.linspace(1, 10, f_len)
        neuron = snn.Neuron("linear", weights, func_weights)

        total_len = neuron.total_len()
        self.assertEqual(total_len, w_len + f_len)

        self.assertEqual(neuron.w_len(), w_len)
        self.assertEqual(neuron.f_len(), f_len)

        _np_tests.assert_array_equal(neuron.get_input_weights(), weights)
        _np_tests.assert_array_equal(neuron.get_func_weights(), func_weights)

        self.assertEqual(neuron.func_name(), "linear")

    def test_neuron_returns_right_output(self):
        weights = _np.array([1.0, 1.0, 1.0])
        inputs = _np.array([3.0, 3.0, 3.0])
        neuron = snn.Neuron("cubic", weights)
        output = neuron.activate(_np.sum(weights * inputs))

        _np_tests.assert_allclose(output, 729.0)

    def  test_neuron_set_weights(self):
        weights = _np.array([1.0, 1.0, 1.0])
        func_weights = _np.array([5.0, 5.0, 5.0, 4.0])

        neuron = snn.Neuron("linear", weights.copy(), func_weights.copy())

        weights.fill(0.0)
        func_weights.fill(25.0)

        neuron.set_input_weights(weights.copy())
        neuron.set_func_weights(func_weights.copy())

        _np_tests.assert_equal(neuron.get_input_weights(), weights)
        _np_tests.assert_equal(neuron.get_func_weights(), func_weights)

    def test_neuron_raises_if_uninitialized_001(self):

        neuron = snn.Neuron("linear")

        self.assertRaises(NameError, neuron.get_input_weights)

    def test_neuron_raises_if_uninitialized_002(self):
        neuron = snn.Neuron("linear", func_weights_count=5)

        self.assertRaises(NameError, lambda: neuron.activate(5))

    def test_neuron_copy(self):
        weights = _np.array([1.0, 1.0, 1.0])
        func_weights = _np.array([5.0, 5.0, 5.0, 4.0])

        neuron = snn.Neuron("linear", weights.copy(), func_weights.copy())
        copy = neuron.copy()

        weights.fill(.0)
        func_weights.fill(3.0)
        neuron.set_input_weights(weights)
        neuron.set_func_weights(func_weights)

        self.assertEqual(_np.equal(neuron.get_input_weights(), copy.get_input_weights()).all(),
                          False)
        self.assertEqual(_np.equal(neuron.get_func_weights(),
                                    copy.get_func_weights()).all(),
                          False)




class TestLayer(unittest.TestCase):

    def test_base_creation(self):
        neurons = list()
        for i in range(0, 10):
            neurons.append(snn.Neuron("linear", _np.random.rand(43)))
        l = snn.Layer()
        l.add_neurons(neurons)
        # Need this to make layer create matrix.
        for neuron in l:
            pass

        self.assertEqual(l.out_len(), 10)
        self.assertEqual(l.in_len(), 43)

    def test_add_neurons_raises(self):
        l = snn.Layer()
        self.assertRaises(ValueError, lambda: l.add_neurons("test"))

    def test_iterable(self):
        neuron_count = 10
        neurons = list()
        for i in range(0, neuron_count):
            neurons.append(snn.Neuron("linear", _np.random.rand(43)))
        l = snn.Layer()
        l.add_neurons(neurons)

        count = 0
        for neuron in l:
            count += 1

        self.assertEqual(count, neuron_count)

    def test_layer_returns_right_output(self):
        neuron_count = 45
        input_len = 56
        neurons = list()
        for i in range(0, neuron_count):
            weights = _np.zeros(input_len, dtype=_np.float64)
            weights.fill(1.0)
            neurons.append(snn.Neuron("linear", weights))

        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        input = _np.zeros(input_len, dtype=_np.float64).reshape(input_len, 1)
        input.fill(2.0)

        expected_output = 2.0 * input_len
        expected_array = _np.zeros(neuron_count, dtype=_np.float64)
        expected_array.fill(expected_output)

        real_array = l.input(input).A1

        _np_tests.assert_allclose(real_array,  expected_array)

    def test_layer_fails_on_bad_input_001(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", _np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        input = _np.linspace(0, 1, 30)

        self.assertRaises(ValueError, lambda: l.input(input))

    def test_layer_fails_on_bad_input_002(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", _np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        input = _np.linspace(0, 1, 45)

        self.assertRaises(ValueError, lambda: l.input(input))

    def test_layer_fails_on_bad_input_003(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", _np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        input = _np.array([])

        self.assertRaises(ValueError, lambda: l.input(input))

    def test_layer_get_weights(self):
        weights = _np.array([5.0, 5.0, 5.0])
        func_weights = _np.array([34.0])

        neurons = list()
        for i in range(0, 150):
            neurons.append(snn.Neuron("linear", weights.copy(), func_weights.copy()))
        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        actual_weights = l.get_weights("input")
        actual_func_weights = l.get_weights("func")
        actual_all_weights = l.get_weights("all")

        _np_tests.assert_equal(actual_weights, 5.0)
        _np_tests.assert_equal(actual_func_weights, 34.0)
        self.assertEqual(actual_all_weights.size, actual_weights.size + actual_func_weights.size)

    def test_layer_set_weights(self):
        weights = _np.array([5.0, 5.0, 5.0])
        func_weights = _np.array([34.0])

        neurons = list()
        for i in range(0, 150):
            neurons.append(
                snn.Neuron("linear", weights.copy(), func_weights.copy()))
        l = snn.Layer()
        l.add_neurons(neurons)
        for n in l:
            pass

        weights = l.get_weights("input")
        weights.fill(0)
        func_weights = l.get_weights("func")
        func_weights.fill(-.4)
        l.set_weights(weights.copy())
        _np_tests.assert_equal(l.get_weights("input"), 0)
        l.set_func_weights(func_weights)
        _np_tests.assert_equal(l.get_weights("func"), -0.4)
