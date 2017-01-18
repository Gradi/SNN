import json as _json
import os.path as _path
import numpy as _np

import snn.utils.fast_nn as _fnn
from snn.core import snn_core as _snn
import snn.network.perceptron as _perceptron


class PerceptronNumpy(_perceptron.Perceptron):
    """
        Implementation of Perceptron using numpy's matrices
        and blas library multiplication.
        PS. Intel MKL, for example, can automatically
        run matrix multiplication using all cores.
    """

    def __init__(self, input_count, error_name="mse"):
        """
        :param input_count:
        :param error_name: Currently only Mean Squared Error is implemented.
        """
        super().__init__(input_count, error_name)
        self.__layers = list()
        self.__input_weights_count = 0
        self.__func_weights_count = 0
        self.__test_inputs = None
        self.__test_results = None
        self.__input_prepared = False
        self.__error = self.__mse

    def set_test_inputs(self, inputs, out):
        assert inputs.shape[1] == self._input_count
        self.__test_inputs = _np.transpose(_np.matrix(inputs))
        self.__test_results = _np.matrix(out)

    def __mse(self):
        self.__input_prepared = True
        nn_results = self.input(self.__test_inputs)
        self.__input_prepared = False
        if nn_results.size != self.__test_results.size:
            raise NameError("Size of net results({}) != "
                            "size of test results({})".
                            format(nn_results.size, self.__test_results.size))

        mse = _np.square((nn_results - self.__test_results))
        mse = _np.sqrt(_np.sum(mse) / nn_results.size)
        return mse

    def error(self, weights=None):
        if self.__test_inputs is None or self.__test_results is None:
            raise NameError("Test data isn't provided.")
        if weights is not None:
            self.set_weights(weights)
        return self.__error()

    def error_input_weights(self, input_weights=None):
        if self.__test_inputs is None or self.__test_results is None:
            raise NameError("Test data isn't provided.")
        if input_weights is not None:
            self.set_weights(input_weights, "input")
        return self.__error()

    def error_func_weights(self, func_weights=None):
        if self.__test_inputs is None or self.__test_results is None:
            raise NameError("Test data isn't provided.")
        if func_weights is not None:
            self.set_weights(func_weights, "func")
        return self.__error()

    def input(self, data):
        if self.__input_prepared:
            result = data
        else:
            result = _np.transpose(_np.matrix(data, copy=False))

        for layer in self.__layers:
            result = layer.input(result)

        if result.size == 1:
            return result.A1[0]
        elif self.net_output_len() == 1:
            return result.A1
        else:
            return result

    def set_weights(self, weights, weights_type="all"):
        if weights_type == "all" or weights_type == "input":
            if weights_type == "all":
                assert weights.size == (self.__input_weights_count +
                                        self.__func_weights_count)
            else:
                assert weights.size == self.__input_weights_count

            total = 0
            for layer in self.__layers:
                weights_count = layer.input_weights_count
                if weights_type == "all":
                    weights_count += layer.func_weights_count
                layer.set_weights(weights[total:total + weights_count])
                total += weights_count
        elif weights_type == "func":
            assert weights.size == self.__func_weights_count
            total = 0
            for layer in self.__layers:
                layer.set_func_weights(
                    weights[total:total + layer.func_weights_count])
                total += layer.func_weights_count
        else:
            raise ValueError("weights_type must be all or input or func.")

    def get_weights(self, weights_type="all"):
        result = _np.array([])
        for layer in self.__layers:
            result = _np.append(result, layer.get_weights(weights_type))
        return result

    def add_layer(self, layer):
        if type(layer) == _snn.Layer:
            layer = layer.copy()
            self.__layers.append(layer)
        elif hasattr(layer, "__iter__") and\
             type(layer[0]) == _snn.Layer:
            for l in layer:
                l = l.copy()
                self.__layers.append(l)
        else:
            raise NameError("Unknown type received. "
                            "Expected Layer or iterable of layer")
        self.reset_weights(False)
        self.__input_weights_count = 0
        self.__func_weights_count  = 0
        for layer in self.layers():
            self.__input_weights_count += layer.input_weights_count
            self.__func_weights_count  += layer.func_weights_count

    def layers(self):
        return self.__layers

    def net_output_len(self):
        if len(self.__layers) == 0:
            raise NameError("Network is empty. Don't know output length yet.")
        else:
            return self.__layers[-1].out_len()

    def reset_weights(self, force=True):
        input_count = self._input_count
        for layer in self.__layers:
            for neuron in layer:
                if force or neuron.w_len() == 0:
                    neuron.set_input_weights(_fnn.rnd_weights(input_count))
                if neuron.f_len() != 0 and (neuron.get_func_weights() is None or force):
                    neuron.set_func_weights(_fnn.rnd_weights(neuron.f_len()))
            input_count = layer.out_len()

    def to_json(self, with_weights=True):
        result = dict()
        result["input_count"] = self._input_count
        result["error_name"]  = self._error_name
        result["layers"] = list()
        for layer in self.__layers:
            layer_dump = list()
            for neuron in layer:
                neuron_dump = dict()
                neuron_dump["func_name"] = neuron.func_name()
                neuron_dump["f_len"] = neuron.f_len()
                if with_weights:
                    neuron_dump["weights"] = neuron.get_input_weights().tolist()
                    if neuron.get_func_weights() is not None:
                        neuron_dump["func_weights"] = neuron.get_func_weights().tolist()
                layer_dump.append(neuron_dump)
            result["layers"].append(layer_dump)
        return _json.dumps(result, indent=4)

    def save_to_file(self, filename, with_weights=True, overwrite=True):
        if not overwrite and _path.exists(filename):
            raise NameError("File {} already exists!".format(filename))
        f = open(filename, "w")
        json_str = self.to_json(with_weights)
        f.write(json_str)
        f.close()

    def copy(self):
        copy = PerceptronNumpy(self._input_count, self._error_name)
        for layer in self.layers():
            copy.add_layer(layer.copy())
        return copy

    def load_from_json(json_str):
        net_dump = _json.loads(json_str)
        input_count = net_dump["input_count"]
        error_name = net_dump["error_name"]
        net = PerceptronNumpy(input_count, error_name)
        for layer_dump in net_dump["layers"]:
            layer = _snn.Layer()
            for neuron_dump in layer_dump:
                if "weights" in neuron_dump:
                    weights = _np.array(neuron_dump["weights"])
                else:
                    weights = None
                if "func_weights" in neuron_dump:
                    func_weights = _np.array(neuron_dump["func_weights"])
                else:
                    func_weights = None
                neuron = _snn.Neuron(neuron_dump["func_name"], weights, func_weights,
                                     neuron_dump["f_len"])
                layer.add_neurons(neuron)
            net.add_layer(layer)
        return net

    def load_from_file(filename):
        f = open(filename, "r")
        json_str = f.read()
        f.close()
        return PerceptronNumpy.load_from_json(json_str)
