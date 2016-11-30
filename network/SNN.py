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

    def __init__(self, input_count, weight_bounds=(-1.0, 1.0),
                 func_bounds=(-1.0, 1.0)):
        self.__layers = list()
        self.__test_inputs = None
        self.__test_results = None
        self.__weight_bounds = weight_bounds
        self.__func_bounds = func_bounds
        self.__input_count = input_count

    def set_test_inputs(self, inputs, out):
        self.__test_inputs = inputs
        self.__test_results = out

    def __mse(self):
        nn_results = self.multi_input(self.__test_inputs)
        if nn_results.size != self.__test_results.size:
            raise NameError("Size of net results({}) != size of test results({})".
                            format(nn_results.size, self.__test_results.size))

        mse = (nn_results - self.__test_results) ** 2
        mse = _np.sqrt(_np.sum(mse) / len(nn_results))
        return mse

    def error(self, weights=None):
        if self.__test_inputs is None or self.__test_results is None:
            raise NameError("Test data isn't provided.")
        if weights is not None:
            self.set_weights(weights)
        return self.__mse()

    def input(self, data):
        result = data
        for layer in self.__layers:
            result = layer.input(result)
        if self.net_output_len() == 1:
            return result[0]
        else:
            return result

    def multi_input(self, data):
        result = list()
        for d in data:
            result.append(self.input(d))
        return _np.array(result)

    def set_weights(self, weights):
        total = 0
        for layer in self.__layers:
            for neuron in layer:
                c_len = neuron.total_len()
                neuron.set_weights(weights[total:total+c_len])
                total += c_len

    def add_layer(self, layer):

        def init_weights(layer):
            in_count = self.__input_count if len(self.__layers) == 0 else \
                       self.__layers[-1].out_len()
            for neuron in layer:
                if neuron.w_len() == 0:
                    w = _fnn.rnd_weights(in_count, self.__weight_bounds)
                    neuron.set_input_weights(w)
                if neuron.get_func_weights() is None and\
                   neuron.f_len() != 0:
                    w = _fnn.rnd_weights(neuron.f_len(), self.__func_bounds)
                    neuron.set_func_weights(w)

        if type(layer) == Layer:
            init_weights(layer)
            self.__layers.append(layer)
        elif hasattr(layer, "__iter__") and\
             type(layer[0]) == _snn.Layer:
            for l in layer:
                init_weights(l)
                self.__layers.append(l)
        else:
            raise NameError("Unknown type received. Expected Layer or iterable of layer")

    def check_bounds(self, weights):
        if self.__weight_bounds is None and self.__func_bounds is None:
            _wrn.warn("Tried to check non existing bounds.")
            return

        total = 0
        for layer in self.__layers:
            for neuron in layer:
                w_len = neuron.w_len()
                f_len = neuron.f_len()

                if self.__weight_bounds is not None:
                    weights[total:total+w_len] = _np.maximum(weights[total:total+w_len], self.__weight_bounds[0])
                    weights[total:total+w_len] = _np.minimum(weights[total:total+w_len], self.__weight_bounds[1])
                if self.__func_bounds is not None:
                    weights[total+w_len:total+w_len+f_len] = _np.maximum(weights[total+w_len:total+w_len+f_len],
                                                                               self.__func_bounds[0])
                    weights[total+w_len:total+w_len+f_len] = _np.minimum(weights[total+w_len:total+w_len+f_len],
                                                                               self.__func_bounds[1])
                total += w_len + f_len

    def get_weights(self):
        result = _np.array([])
        for layer in self.__layers:
            for neuron in layer:
                result = _np.append(result, neuron.get_weights())
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