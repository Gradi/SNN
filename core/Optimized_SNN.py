import numpy as _np

from SNN.core.Optimized_Layer import  OptimizedLayer as _Layer
from SNN.network import SNN as _snn


class OptSNN:

    def __init__(self, snn):
        self.__weights_bounds = snn.get_weights_bounds()
        self.__func_bounds = snn.get_func_bounds()
        self.__layers = list()
        self.__test_inputs = None
        self.__test_outputs = None
        self.__total_weights_count = 0
        self.__input_is_prepared = False
        for layer in snn.layers():
            new_l = _Layer(layer)
            self.__total_weights_count += new_l.weights_count
            self.__layers.append(new_l)

    def set_test_data(self, input, out):
        """
            Set test inputs and according to inputs outputs.
            This inputs and outputs will be used for
            computation of an error function.
        :param input: numpy array of inputs like [x1, x2, x3]
        :param out: numpy array of outputs like [y1, y2, y3] where
                    y_i is a result of neural network when input was x_i
        """
        self.__test_inputs = _np.matrix(_np.zeros(input.shape))
        for ri in range(0, len(input)):
            self.__test_inputs[ri] = input[ri]
        self.__test_inputs = self.__test_inputs.T
        self.__test_outputs = out

    def error(self, weights=None):
        if weights is not None:
            self.set_weights(weights)
        return self.__mse()

    def set_weights(self, weights):
        assert weights.size == self.__total_weights_count
        total_weights = 0
        for layer in self.__layers:
            layer.set_weights(weights[total_weights:
                                      total_weights + layer.weights_count])
            total_weights += layer.weights_count

    def get_weights(self):
        res = _np.array([])
        for layer in self.__layers:
            res = _np.append(res, layer.get_weights())
        assert res.size == self.__total_weights_count
        return res

    def input(self, x):
        if not self.__input_is_prepared:
            x = _np.matrix(x).T
        for layer in self.__layers:
            x = layer.input(x)
        if x.size == 1:
            return x.A1[0]
        else:
            return x

    def __mse(self):
        if self.__test_inputs is None and \
           self.__test_outputs is None:
            raise NameError("Test data isn't provided.")

        self.__input_is_prepared = True
        nn_results = self.input(self.__test_inputs)
        self.__input_is_prepared = False
        assert nn_results.size == self.__test_outputs.size

        mse = _np.square((nn_results - self.__test_outputs))
        mse = _np.sqrt(_np.sum(mse) / nn_results.size)
        return mse

    def to_simple_snn(self):
        res = _snn.SNN(0, self.__weights_bounds, self.__func_bounds)
        for l in self.__layers:
            res.add_layer(l.back_to_layer())
        return res
