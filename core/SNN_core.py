import numpy as _np

from core import neuron_functions as _nf


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

    def __init__(self, func_name, weights, func_weights=None):
        self.__weights = weights
        self.__func_weights = func_weights
        self.__func_name = func_name
        self.__func = None
        self.__input_sum = 0.0

    def input(self, inputs):
        if inputs.size != self.w_len():
            raise NameError("Len of inputs({}) != len of weights({})."
                            .format(inputs.size, self.w_len()))
        if self.__func is None:
            self.__func = _nf.get(self.__func_name)

        self.__input_sum = _np.sum(inputs * self.__weights)
        if self.__func_weights is None:
            return self.__func(self.__input_sum)
        else:
            return self.__func(self.__input_sum, self.__func_weights)

    def get_weights(self):
        if self.__func_weights is None:
            return _np.array(self.__weights)
        else:
            result = _np.zeros(self.total_len(), dtype=_np.float64)
            result[0:self.w_len()] = _np.array(self.__weights)
            result[self.w_len():] = _np.array(self.__func_weights)
            return result

    def set_weights(self, weights):
        if self.total_len() != weights.size:
            raise NameError("Len of new weights({}) != len of old weights({})".
                            format(weights.size, self.total_len()))

        if self.__func_weights is None:
            self.__weights = weights
        else:
            w_len = self.w_len()
            f_len = self.f_len()
            self.__weights = weights[0:w_len]
            self.__func_weights = weights[w_len:w_len + f_len]

    def w_len(self):
        return self.__weights.size

    def f_len(self):
        return 0 if self.__func_weights is None else self.__func_weights.size

    def total_len(self):
        return self.w_len() + self.f_len()

    def func_name(self):
        return self.__func_name

    def get_input_weights(self):
        return _np.array(self.__weights)

    def get_func_weights(self):
        return None if self.__func_weights is None else _np.array(self.__func_weights)


class Layer:

    def __init__(self):
        self.__neurons = list()

    def add_neurons(self, neuron):
        if hasattr(neuron, "__iter__"):
            for n in neuron:
                self.__neurons.append(n)
        else:
            self.__neurons.append(neuron)

    def __iter__(self):
        return iter(self.__neurons)

    def input(self, input_data):
        results = _np.zeros(self.out_len())
        i = 0
        for neuron in self:
            output = neuron.input(input_data)
            results[i] = output
            i += 1
        return results

    def out_len(self):
        return len(self.__neurons)

    def in_len(self):
        inputs_len = list()
        for neuron in self:
            inputs_len.append(neuron.w_len())
        inputs_len = _np.array(inputs_len)
        if not _np.equal(inputs_len, inputs_len[0]).all():
            raise NameError("Neurons have different input length!")
        else:
            return inputs_len[0]
