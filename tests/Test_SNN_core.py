import unittest

import numpy as np
import numpy.testing as np_tests

from core import SNN_core as snn


class TestFlatArrayContainer(unittest.TestCase):

    def setUp(self):
        self.dtype = np.int32
        a1 = np.array([i for i in range(1, 6)])
        a2 = np.array([i for i in range(6, 11)])
        a3 = np.array([i for i in range(11, 16)])
        self.individual_arrs = [a1, a2, a3]

        self.flat_arr_expected = np.array([], dtype=self.dtype)
        for arr in self.individual_arrs:
            self.flat_arr_expected = np.append(self.flat_arr_expected, arr)

    def test_creation_successfull(self):
        real = snn.FlatArrayContainer(dtype=self.dtype)
        for arr in self.individual_arrs:
            real.add_array(arr)

        np_tests.assert_array_equal(real, self.flat_arr_expected)

    def test_return_equal_original(self):
        real = snn.FlatArrayContainer(dtype=self.dtype)
        for arr in self.individual_arrs:
            real.add_array(arr)

        returned_arrs = list()
        for arr in real.generator():
            returned_arrs.append(arr)

        self.assertEqual(len(returned_arrs), len(self.individual_arrs))
        for i in range(0, len(returned_arrs)):
            np_tests.assert_array_equal(returned_arrs[i], self.individual_arrs[i])

    def test_can_do_math_operations(self):
        a = snn.FlatArrayContainer(self.dtype)
        b = snn.FlatArrayContainer(self.dtype)
        for arr in self.individual_arrs:
            a.add_array(arr)
            b.add_array(arr)

        c = a + b

        self.assertEqual(type(c), snn.FlatArrayContainer)

        np_tests.assert_array_equal(c, self.flat_arr_expected + self.flat_arr_expected)
        np_tests.assert_array_equal(a, self.flat_arr_expected)
        np_tests.assert_array_equal(b, self.flat_arr_expected)

    def test_can_use_in_functions(self):
        a = snn.FlatArrayContainer(self.dtype)
        for arr in self.individual_arrs:
            a.add_array(arr)


        def f(x):
            return x ** 2

        b = f(a)

        self.assertEquals(type(b), snn.FlatArrayContainer)

        np_tests.assert_array_equal(b, self.flat_arr_expected ** 2)
        np_tests.assert_array_equal(a, self.flat_arr_expected)

    def test_copy_change_doesnt_affect_original(self):
        a = snn.FlatArrayContainer(self.dtype)
        for arr in self.individual_arrs:
            a.add_array(arr)
        copy = a.copy()
        for i in range(0, len(copy)):
            copy[i] = copy[i] ** 2

        np_tests.assert_array_equal(a, self.flat_arr_expected)
        np_tests.assert_array_equal(copy, self.flat_arr_expected ** 2)

    def test_len_works(self):
        a = snn.FlatArrayContainer(self.dtype)
        for arr in self.individual_arrs:
            a.add_array(arr)

        self.assertEquals(len(a), len(self.flat_arr_expected))

    def test_cant_add_other_dtype_arr(self):
        a = snn.FlatArrayContainer(self.dtype)
        a.add_array(self.flat_arr_expected)
        b = np.array([x for x in np.linspace(0, 1, 100, dtype=np.float64)], dtype=np.float64)

        self.assertRaises(NameError, lambda: a.add_array(b))

    def test_internal_replace(self):
        x = np.linspace(1, 100, 100, dtype=np.float64)
        f = snn.FlatArrayContainer(np.float64)
        f.add_array(x)

        np_tests.assert_array_equal(f, x)

        for i in range(0, len(x)):
            x[i] = x[i] * x[i]

        f.internal_replace(x)
        np_tests.assert_array_equal(f, x)

        x[0] = -1.0
        res = np.equal(x, f)
        self.assertEquals(res[0], False)
        self.assertEquals(np.all(res[1:]), True)

    def test_copy_is_same_type(self):
        a = snn.FlatArrayContainer(self.dtype)
        a.add_array(self.flat_arr_expected)
        copy = a.copy()

        self.assertEqual(type(copy), type(a))


class TestNeuron(unittest.TestCase):

    def test_base_creation(self):
        w_len = 10
        f_len = 1
        weights = np.linspace(1, 10, w_len)
        func_weights = np.linspace(1, 10, f_len)
        n = snn.Neuron("linear", weights, func_weights)

        total_len = n.total_len()
        self.assertEqual(total_len, w_len + f_len)

        self.assertEqual(n.w_len(), w_len)
        self.assertEqual(n.f_len(), f_len)

        all_weights = np.array([])
        all_weights = np.append(all_weights, weights)
        all_weights = np.append(all_weights, func_weights)

        np_tests.assert_array_equal(n.get_weights(), all_weights)
        np_tests.assert_array_equal(n.get_input_weights(), weights)
        np_tests.assert_array_equal(n.get_func_weights(), func_weights)

        self.assertEqual(n.func_name(), "linear")

    def test_neuron_returs_right_output(self):
        weights = np.array([1.0, 1.0, 1.0])
        inputs = np.array([3.0, 3.0, 3.0])
        n = snn.Neuron("linear", weights)
        output = n.input(inputs)

        np_tests.assert_allclose(output, 9.0)

    def test_neuron_fails_on_bad_inputs_001(self):
        weights = np.linspace(1, 10, 10)
        n = snn.Neuron("linear", weights)
        inputs = np.linspace(1, 10, 50)

        self.assertRaises(NameError, lambda: n.input(inputs))

    def test_neuron_fails_on_bad_inputs_002(self):
        weights = np.linspace(1, 10, 50)
        n = snn.Neuron("linear", weights)
        inputs = np.linspace(1, 10, 10)

        self.assertRaises(NameError, lambda: n.input(inputs))

    def test_neuron_fails_on_bad_inputs_003(self):
        weights = np.linspace(1, 10, 10)
        n = snn.Neuron("linear", weights)
        inputs = list()
        for i in range(0, 10):
            inputs.append(np.random.rand(10))
        inputs = np.array(inputs)

        self.assertRaises(NameError, lambda: n.input(inputs))

    def test_neuron_returns_all_weights(self):
        weights = np.linspace(1, 10, 10)
        neuron = snn.Neuron("linear", weights[0:5], weights[5:10])
        real_weights = neuron.get_weights()
        np_tests.assert_array_equal(real_weights, weights)


class TestLayer(unittest.TestCase):

    def test_base_creation(self):
        neurons = list()
        for i in range(0, 10):
            neurons.append(snn.Neuron("linear", np.random.rand(43)))
        l = snn.Layer()
        l.add_neurons(neurons)

        self.assertEqual(l.out_len(), 10)
        self.assertEqual(l.in_len(), 43)

        for neuron in l:
            self.assertEqual(type(neuron), snn.Neuron)

    def test_layer_returns_right_output(self):
        neuron_count = 45
        input_len = 56
        neurons = list()
        for i in range(0, neuron_count):
            weights = np.zeros(input_len, dtype=np.float64)
            weights.fill(1.0)
            neurons.append(snn.Neuron("linear", weights))

        l = snn.Layer()
        l.add_neurons(neurons)

        input = np.zeros(input_len, dtype=np.float64)
        input.fill(2.0)

        expected_output = 2.0 * input_len
        expected_array = np.zeros(neuron_count, dtype=np.float64)
        expected_array.fill(expected_output)

        real_array = l.input(input)

        np_tests.assert_allclose(real_array,  expected_array)

    def test_layer_faild_on_bad_input_001(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)

        input = np.linspace(0, 1, 30)

        self.assertRaises(NameError, lambda: l.input(input))

    def test_layer_faild_on_bad_input_002(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)

        input = np.linspace(0, 1, 45)

        self.assertRaises(NameError, lambda: l.input(input))

    def test_layer_faild_on_bad_input_003(self):
        neurons = list()
        for i in range(0, 148):
            neurons.append(snn.Neuron("linear", np.random.rand(44)))

        l = snn.Layer()
        l.add_neurons(neurons)

        input = np.array([])

        self.assertRaises(NameError, lambda: l.input(input))
