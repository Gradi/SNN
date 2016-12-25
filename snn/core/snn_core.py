import numpy as _np

from snn.core import neuron_functions as _nf


class FlatArrayContainer(_np.ndarray):
    """
        Currently this class is not used.
        Maybe i should delete it.
    """

    def __new__(cls, dtype):
        obj = super(FlatArrayContainer, cls).__new__(cls, shape=0, dtype=dtype, buffer=None)
        obj.arrays_indexes = _np.array([0], dtype=_np.uint32)
        return obj

    def add_array(self, array):
        if array.dtype != self.dtype:
            raise NameError("dtype of input array({}) != my own dtype({})".format(array.dtype, self.dtype))

        last_ar = self.arrays_indexes[-1]
        new_ar = len(array) + last_ar
        self.arrays_indexes = _np.append(self.arrays_indexes, new_ar)
        self.resize(new_ar, refcheck=False)
        self[last_ar:new_ar] = _np.array(array, dtype=self.dtype)

    def generator(self):
        for i in range(0, len(self.arrays_indexes) - 1):
            start = self.arrays_indexes[i]
            end = self.arrays_indexes[i+1]
            yield self[start:end]

    def copy(self):
        copy = FlatArrayContainer(self.dtype)
        for array in self.generator():
            copy.add_array(array)
        return copy

    def internal_replace(self, array, copy=True):
        if len(self) != len(array):
            raise NameError("Length of new internal array({}) != my current len({})".format(len(array), len(self)))
        if self.dtype != array.dtype:
            raise NameError("dtype  of new internal array({}) != my current dtype({})".format(array.dtype, self.dtype))
        if copy:
            self.data = _np.array(array).data
        else:
            self.data = array.data


class Neuron:

    def __init__(self, func_name, weights=None, func_weights=None,
                 func_weights_count=0):
        self.__weights = weights
        self.__func_weights = func_weights
        self.__func_name = func_name
        self.__func = _nf.get(func_name)
        if func_weights is not None:
            self.__func_weights_count = len(func_weights)
        else:
            self.__func_weights_count = func_weights_count

    def activate(self, input_sum):
        if self.__func_weights is not None:
            return self.__func(input_sum, self.__func_weights)
        elif self.__func_weights_count != 0:
            raise NameError("Non initialized neuron: func weights isn't set!")
        else:
            return self.__func(input_sum)

    def w_len(self):
        return 0 if self.__weights is None else self.__weights.size

    def f_len(self):
        return self.__func_weights_count

    def total_len(self):
        return self.w_len() + self.f_len()

    def func_name(self):
        return self.__func_name

    def get_input_weights(self):
        if self.__weights is None:
            raise NameError("Attempt to retrieve weights from non initialized neuron!")
        else:
            return _np.array(self.__weights)

    def get_func_weights(self):
        return None if self.__func_weights is None else _np.array(self.__func_weights)

    def set_input_weights(self, weights):
        self.__weights = weights

    def set_func_weights(self, weights):
         self.__func_weights = weights

    def copy(self):
        weights = None
        func_weights = None
        if self.__weights is not None:
            weights = _np.array(self.__weights)
        if self.__func_weights is not None:
            func_weights = _np.array(self.__func_weights)
        return Neuron(self.func_name(), weights, func_weights, self.__func_weights_count)


class Layer:

    def __init__(self):
        self.__neurons = list()
        self.weights_count = 0
        self.__W = None

    def add_neurons(self, neuron):
        if hasattr(neuron, "__iter__") and\
           type(neuron[0]) == Neuron:
            for n in neuron:
                self.__neurons.append(n)
        elif type(neuron) == Neuron:
            self.__neurons.append(neuron)
        else:
            raise ValueError("Expected neuron or iterator of neurons.")

    def __iter__(self):
        if self.__W is None:
            return iter(self.__neurons)
        else:
            return self.__neuron_generator()

    def __neuron_generator(self):
        for ri in range(0, self.out_len()):
            weights = _np.array(self.__W[ri].A1)
            neuron = self.__neurons[ri]
            neuron.set_input_weights(weights)
            yield neuron

    def input(self, input):
        result = self.__W * input
        for ri in range(0, len(self.__neurons)):
            result[ri] = self.__neurons[ri].activate(result[ri])
        return result

    def out_len(self):
        if self.__W is None:
            raise NameError("Layer is not initialized.")
        else:
            return self.__W.shape[0]

    def in_len(self):
        if self.__W is None:
            raise NameError("Layer is not initialized.")
        else:
            return self.__W.shape[1]

    def copy(self):
        copy = Layer()
        for neuron in self:
            copy.add_neurons(neuron.copy())
        return copy

    def get_weights(self):
        res = _np.array(self.__W.A1)
        for neuron in self.__neurons:
            if neuron.f_len() != 0:
                res = _np.append(res, neuron.get_func_weights())
        return res

    def set_weights(self, weights):
        W = weights[0:self.__W.size].reshape(self.__W.shape)
        self.__W = _np.matrix(W, copy=False)
        if self.__W.size != weights.size:
            total = 0
            func_weights = weights[self.__W.size:]
            for neuron in self.__neurons:
                f_len = neuron.f_len()
                if f_len != 0:
                    neuron.set_func_weights(func_weights[total:total + f_len])
                    total += f_len

    def init_layer(self):
        self.__W = _np.matrix(self.__neurons[0].get_input_weights())
        self.weights_count += self.__neurons[0].total_len()
        for neuron in self.__neurons[1:]:
            self.__W = _np.vstack((self.__W, neuron.get_input_weights()))
            self.weights_count += neuron.total_len()

