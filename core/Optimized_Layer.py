import numpy as _np
import SNN.core.neuron_functions as _nf
import SNN.core.SNN_core as _snn


class OptimizedLayer:

    def __init__(self, layer):
        self.weights_count = 0
        self.__rows = layer.out_len()
        self.__cols = layer.in_len()
        self.__functions = list()
        self.__W = _np.matrix(_np.zeros((self.__rows, self.__cols)))
        ri = 0

        for neuron in layer:
            self.__W[ri] = neuron.get_input_weights()
            ri += 1
            func = dict()
            func["func_name"] = neuron.func_name()
            func["f"] = _nf.get(neuron.func_name())
            func["fargs"] = neuron.get_func_weights()
            self.__functions.append(func)
            self.weights_count += neuron.total_len()

        assert ri == self.__rows
        assert ri == len(self.__functions)

    def input(self, x):
        res = self.__W * x
        for i in range(0, len(self.__functions)):
            f = self.__functions[i]["f"]
            fargs = self.__functions[i]["fargs"]
            if fargs is not None:
                res[i] = f(res[i], fargs)
            else:
                res[i] = f(res[i])
        return res

    def get_weights(self):
        res = self.__W.A1
        for func in self.__functions:
            if func["fargs"] is not None:
                res = _np.append(res, func["fargs"])
        return res

    def set_weights(self, weights):
        for ri in range(0, self.__rows):
            self.__W[ri] = weights[ri*self.__cols:ri*self.__cols + self.__cols]
        if self.__rows * self.__cols != self.weights_count:
            total_f_len = 0
            offset = self.__rows * self.__cols
            for func in self.__functions:
                if func["fargs"] is not None:
                  f_len = len(func["fargs"])
                  func["fargs"] = weights[offset + total_f_len:
                                          offset + total_f_len + f_len]
                  total_f_len += f_len

    def out_len(self):
        return self.__W.shape[0]

    def back_to_layer(self):
        layer = _snn.Layer()
        for ri in range(0, self.__rows):
            weights = self.__W[ri].A1
            func_weights = self.__functions[ri]["fargs"]
            func_name = self.__functions[ri]["func_name"]
            neuron = _snn.Neuron(func_name, weights, func_weights)
            layer.add_neurons(neuron)
        return layer
