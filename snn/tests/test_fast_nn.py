import unittest
import numpy as _np
import numpy.testing as _np_test
import snn.utils.fast_nn as _fnn
import snn

class Test_fast_nn(unittest.TestCase):

    def test_make_neurons_001(self):
        neuron_count = 5
        func_name = "linear"

        neurons = _fnn.make_neurons(func_name, neuron_count)

        self.assertEquals(len(neurons), neuron_count)
        for neuron in neurons:
            self.assertEquals(neuron.func_name(), func_name)
            self.assertEquals(neuron.f_len(), 0)
            self.assertEquals(type(neuron), snn.Neuron)

    def test_make_neurons_002(self):
        neuron_count = 5
        func_weights_count = 2
        func_name = ["linear", "cubic"]

        neurons = _fnn.make_neurons(func_name, neuron_count, func_weights_count)

        self.assertEquals(len(neurons), neuron_count)
        for i in range(0, len(neurons)):
            self.assertEquals(neurons[i].f_len(), func_weights_count)
            self.assertEquals(type(neurons[i]), snn.Neuron)
            if i < 1:
                self.assertEquals(neurons[i].func_name(), func_name[0])
            else:
                self.assertEquals(neurons[i].func_name(), func_name[1])

    def test_make_layer_001(self):
        neuron_count = 5
        func_name = "linear"

        layer = _fnn.make_layer(func_name, neuron_count)

        self.assertEquals(type(layer), snn.Layer)
        count = 0
        for neuron in layer:
            self.assertEquals(neuron.func_name(), func_name)
            self.assertEquals(neuron.f_len(), 0)
            count += 1
        self.assertEquals(count, neuron_count)


    def test_rnd_weights(self):
        weights = _fnn.rnd_weights(50000)
        min = _np.min(weights)
        max = _np.max(weights)

        _np_test.assert_allclose(min, -1.0, rtol=1e-3)
        _np_test.assert_allclose(max, 1.0, rtol=1e-3)
