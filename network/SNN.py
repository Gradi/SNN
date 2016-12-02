import json as _json
import os.path as _path
import warnings as _wrn
import numpy as _np

import SNN.utils.fast_nn as _fnn
from SNN.core import SNN_core as _snn
from SNN.core import Optimized_SNN as _opt_snn

Layer  = _snn.Layer
Neuron = _snn.Neuron
OptSNN = _opt_snn.OptSNN


class SNN:

    def __init__(self, input_count,
                 weight_bounds=(-1.0, 1.0),
                 func_bounds=(-1.0, 1.0)):
        self.__layers = list()
        self.__weights_count = 0
        self.__test_inputs = None
        self.__test_results = None
        self.__weight_bounds = weight_bounds
        self.__func_bounds = func_bounds
        self.__input_count = input_count
        self.__input_prepared = False

    def set_test_inputs(self, inputs, out):
        self.__test_inputs = _np.matrix(inputs).T
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
        return self.__mse()

    def input(self, data):
        if self.__input_prepared:
            result = data
        else:
            result = _np.matrix(data, copy=False).T

        for layer in self.__layers:
            result = layer.input(result)

        if result.size == 1:
            return result.A1[0]
        elif self.net_output_len() == 1:
            return result.A1
        else:
            return result

    def set_weights(self, weights):
        assert weights.size == self.__weights_count

        total = 0
        for layer in self.__layers:
            layer.set_weights(weights[total:total + layer.weights_count])
            total += layer.weights_count

    def add_layer(self, layer):
        if type(layer) == Layer:
           self.__init_weights(layer)
           self.__weights_count += layer.weights_count
           self.__layers.append(layer)
        elif hasattr(layer, "__iter__") and\
             type(layer[0]) == Layer:
            for l in layer:
                self.__init_weights(l)
                self.__weights_count += l.weights_count
                self.__layers.append(l)
        else:
            raise NameError("Unknown type received. "
                            "Expected Layer or iterable of layer")

    def __init_weights(self, layer):
        input_count = self.__input_count if len(self.__layers) == 0\
                      else self.__layers[-1].out_len()
        for neuron in layer:
            if neuron.w_len() == 0:
                w = _fnn.rnd_weights(input_count, self.__weight_bounds)
                neuron.set_input_weights(w)
            if neuron.get_func_weights is None and\
               neuron.f_len() != 0:
                f = _fnn.rnd_weights(neuron.f_len(), self.__func_bounds)
                neuron.set_func_weights(f)
        layer.init_layer()

    def get_weights(self):
        result = _np.array([])
        for layer in self.__layers:
            result = _np.append(result, layer.get_weights())
        return result

    def layers(self):
        return self.__layers

    def net_output_len(self):
        if len(self.__layers) == 0:
            return NameError("Network is empty. Don't know output length yet.")
        else:
            return self.__layers[-1].out_len()

    def to_json(self, with_weights=True):
        result = dict()
        result["weight_bounds"] = self.__weight_bounds
        result["func_bounds"] = self.__func_bounds
        result["input_count"] = self.__input_count
        result["layers"] = list()
        for layer in self.__layers:
            layer_dump = list()
            for neuron in layer:
                neuron_dump = dict()
                neuron_dump["func_name"] = neuron.func_name()
                neuron_dump["w_len"] = neuron.w_len()
                neuron_dump["f_len"] = neuron.f_len()
                if with_weights:
                    neuron_dump["weights"] = neuron.get_input_weights().tolist()
                    if neuron.get_func_weights() is not None:
                        neuron_dump["func_weights"] = neuron.get_func_weights().tolist()
                layer_dump.append(neuron_dump)
            result["layers"].append(layer_dump)
        return _json.dumps(result, indent=4)

    def save_to_file(self, filename, with_weights=True, overwrite=True):
        if not overwrite:
            if _path.exists(filename):
                raise NameError("File {} already exists!".format(filename))
        f = open(filename, "w")
        json_str = self.to_json(with_weights)
        f.write(json_str)
        f.close()

    def get_weights_bounds(self):
        return self.__weight_bounds

    def get_func_bounds(self):
        return self.__func_bounds

    def copy(self):
        copy = SNN(self.__input_count, self.__weight_bounds, self.__func_bounds)
        for layer in self.layers():
            copy.add_layer(layer.copy())
        return copy


def load_from_json(json_str):
    net_dump = _json.loads(json_str)
    weight_bounds = net_dump["weight_bounds"]
    func_bounds = net_dump["func_bounds"]
    input_count = net_dump["input_count"]
    net = SNN(input_count, weight_bounds, func_bounds)
    for layer_dump in net_dump["layers"]:
        layer = Layer()
        for neuron_dump in layer_dump:
            if "weights" in neuron_dump:
                weights = _np.array(neuron_dump["weights"])
            else:
                weights = None
            if "func_weights" in neuron_dump:
                func_weights = _np.array(neuron_dump["func_weights"])
            else:
                func_weights = None
            neuron = Neuron(neuron_dump["func_name"], weights, func_weights,
                            neuron_dump["f_len"])
            layer.add_neurons(neuron)
        net.add_layer(layer)
    return net


def load_from_file(filename):
    f = open(filename, "r")
    json_str = f.read()
    f.close()
    return load_from_json(json_str)
