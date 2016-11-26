import json as _json
import os.path as _path
import warnings as _wrn
import numpy as _np

import utils.fast_nn as _fnn
from core import SNN_core as _snn
from core import Optimized_SNN as _opt_snn

Layer  = _snn.Layer
Neuron = _snn.Neuron
OptSNN = _opt_snn.OptSNN


class SNN:

    def __init__(self, weight_bounds=None, func_bounds=None):
        self.__layers = list()
        self.__test_inputs = None
        self.__test_results = None
        self.__weight_bounds = weight_bounds
        self.__func_bounds = func_bounds

    def set_test_inputs(self, test_inputs, test_results):
        self.__test_inputs = test_inputs
        self.__test_results = test_results

    def __mse(self):
        nn_results = list()
        for test_input in self.__test_inputs:
            nn_results.append(self.input(test_input))
        nn_results = _np.array(nn_results)
        if nn_results.size != self.__test_results.size:
            raise NameError("Size of net results({}) != size of test results({})".
                            format(nn_results.size, self.__test_results.size))

        total_sum = _np.zeros(self.net_output_len())
        for i in range(0, len(nn_results)):
            total_sum += (nn_results[i] - self.__test_results[i]) ** 2
        total_sum = _np.sqrt(_np.sum(total_sum) / len(nn_results))
        return total_sum

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

        def check_layer(layer):
            if len(self.__layers) == 0:
                return
            out_len = self.__layers[-1].out_len()
            in_len = layer.in_len()
            if out_len != in_len:
                raise NameError("Len of input of new layer({}) !="
                                " output len of previous layer({})".
                                format(in_len, out_len))

        if type(layer) == Layer:
            check_layer(layer)
            self.__layers.append(layer)
        elif hasattr(layer, "__iter__"):
            for l in layer:
                check_layer(l)
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
        return self.__layers[-1].out_len()

    def to_json(self, with_weights=True):
        result = dict()
        if self.__weight_bounds is not None:
            result["weight_bounds"] = self.__weight_bounds
        if self.__func_bounds is not None:
            result["func_bounds"] = self.__func_bounds
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


def load_from_json(json_str):
    net_dump = _json.loads(json_str)
    if "weight_bounds" in net_dump:
        weight_bounds = net_dump["weight_bounds"]
    else:
        weight_bounds = None
    if "func_bounds" in net_dump:
        func_bounds = net_dump["func_bounds"]
    else:
        func_bounds = None
    net = SNN(weight_bounds, func_bounds)
    for layer_dump in net_dump["layers"]:
        layer = Layer()
        for neuron_dump in layer_dump:
            if "weights" in neuron_dump:
                weights = _np.array(neuron_dump["weights"])
            else:
                if weight_bounds is not None:
                    weights = _fnn.rnd_weights(neuron_dump["w_len"], net_dump["weight_bounds"])
                else:
                    weights = _fnn.rnd_weights(neuron_dump["w_len"])
            if "func_weights" in neuron_dump:
                func_weights = _np.array(neuron_dump["func_weights"])
            elif neuron_dump["f_len"] != 0:
                if func_bounds is not None:
                    func_weights = _fnn.rnd_weights(neuron_dump["f_len"], net_dump["func_bounds"])
                else:
                    func_weights = _fnn.rnd_weights(neuron_dump["f_len"])
            else:
                func_weights = None
            neuron = Neuron(neuron_dump["func_name"], weights, func_weights)
            layer.add_neurons(neuron)
        net.add_layer(layer)
    return net


def load_from_file(filename):
    f = open(filename, "r")
    json_str = f.read()
    f.close()
    return load_from_json(json_str)
