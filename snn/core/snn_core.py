import numpy as _np

from snn.core import neuron_functions as _nf


class Neuron:

    def __init__(self,
                 func_name,
                 weights=None,
                 func_weights=None,
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
            raise NameError("Non initialized neuron: func weights aren't set!")
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
        self.input_weights_count = 0
        self.func_weights_count  = 0
        self.__W = None
        self.__F = None
        self.__function = None

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
        return self.__neuron_generator()

    def __neuron_generator(self):
        # When W is None it means that neurons might not have
        # input weights initialized and then we must just return
        # neurons.
        # But when W is not None first of all we must update input weights
        # of neurons before we return them.
        # In any case after we "yielded" all neurons
        # we must update matrix W, because someone could change neurons weights
        # during iteration.

        if self.__W is None:
            for neuron in self.__neurons:
                yield neuron
        else:
            for ri in range(0, self.out_len()):
                weights = _np.array(self.__W[ri].A1)
                neuron = self.__neurons[ri]
                neuron.set_input_weights(weights)
                yield neuron
        self.__update_matrix()
        self.__optimize_neurons()

    def input(self, input):
        result = self.__W * input
        if self.__function is not None:
            if self.__F is not None:
                assert result.shape[0] == self.__F.size
                result = _np.array(result, copy=False)
                result = self.__function(result, self.__F.reshape((result.shape[0], 1)))
                return _np.matrix(result)
            else:
                return self.__function(result)
        else:
            for ri in range(0, self.out_len()):
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

    def get_weights(self, weights_type="all"):
        func_weights = _np.array([])
        for neuron in self:
            if neuron.f_len() != 0:
                func_weights = _np.append(func_weights, neuron.get_func_weights())
        if weights_type == "all":
            res = _np.array(self.__W.A1)
            res = _np.append(res, func_weights)
            return res
        elif weights_type == "input":
            return _np.array(self.__W.A1)
        elif weights_type == "func":
            return func_weights
        else:
            raise ValueError("Type must be all or input or func.")

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

    def set_func_weights(self, weights):
        total = 0
        for neuron in self.__neurons:
            f_len = neuron.f_len()
            if f_len != 0:
                neuron.set_func_weights(weights[total:total + f_len])
                total += f_len

    def __update_matrix(self):
        try:
            self.input_weights_count = 0
            self.func_weights_count  = 0
            self.__W = _np.matrix(self.__neurons[0].get_input_weights())
            self.input_weights_count += self.__neurons[0].w_len()
            self.func_weights_count  += self.__neurons[0].f_len()
            for neuron in self.__neurons[1:]:
                self.__W = _np.vstack((self.__W, neuron.get_input_weights()))
                self.input_weights_count += neuron.w_len()
                self.func_weights_count  += neuron.f_len()
        except:
            self.__W = None
            # Well. It seems that neurons don't have weights yet.

    def __optimize_neurons(self):
        if self.__neurons_can_be_optimized():
            self.__function = _nf.get(self.__neurons[0].func_name())
            if self.func_weights_count:
                self.__F = _np.array([], dtype=_np.float)
                for neuron in self.__neurons:
                    self.__F = _np.append(self.__F, neuron.get_)

    def __neurons_can_be_optimized(self):
        # If all neurons have the same function and func_weights
        # are zero or one then we can instead of calling each neuron's
        # activation function call it once with x being input in current layer
        # and F being all func weights in one array.
        # Because numpy can run function element wisely and because len of F
        # equal to len of input (or zero if every neuron doesn't have func_weights)
        func_name = self.__neurons[0].func_name()
        for neuron in self.__neurons:
            if func_name != neuron.func_name():
                return False
            if neuron.f_len() != 0 and neuron.f_len() != 1:
                return False
