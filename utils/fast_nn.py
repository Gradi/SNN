from numpy.random import rand as _rnd

import SNN as _snn


def make_neurons(func_name, neuron_count=-1, func_weights_count=0):
    result = list()
    if neuron_count == -1:
        if type(func_name) == str:
            neuron_count = 1
        elif hasattr(func_name, "__iter__") and\
             type(func_name[0]) == str:
            neuron_count = len(func_name)
        else:
            raise ValueError("Expected type str or list of str")

    for i in range(0, neuron_count):
        if type(func_name) == str:
            neuron = _snn.Neuron(func_name, func_weights_count=func_weights_count)
        elif hasattr(func_name, "__iter__") and\
             type(func_name[0]) == str:
            index = -1 if i >= len(func_name) else i
            neuron = _snn.Neuron(func_name[index], func_weights_count=func_weights_count)
        else:
            raise ValueError("Expected type str or list of str")
        result.append(neuron)

    if neuron_count == 1:
        return result[0]
    else:
        return result


def make_layer(func_name, neuron_count=-1, func_weights_count=0):
    neurons = make_neurons(func_name, neuron_count, func_weights_count)
    result = _snn.Layer()
    result.add_neurons(neurons)
    return result


def rnd_weights(num, bounds=(-1.0, 1.0)):
    return bounds[0] + _rnd(num) * (bounds[1] - bounds[0])
