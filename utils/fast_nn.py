from numpy.random import rand as _rnd
import SNN_core as _snn


def make_neurons(func_name, input_count, neuron_count, func_weights_count=0,
                 weight_bounds=(-1.0, 1.0), func_bounds=(-1.0, 1.0)):
    result = list()
    for i in range(0, neuron_count):
        weights = rnd_weights(input_count, weight_bounds)
        func_weights = None
        if func_weights_count:
            func_weights = rnd_weights(func_weights_count, func_bounds)
        result.append(_snn.Neuron(func_name, weights, func_weights))
    if neuron_count == 1:
        return result[0]
    else:
        return result


def make_layer(func_name, layer_input, neuron_count, func_weights_count=0,
               weight_bounds=(-1.0, 1.0), func_bounds=(-1.0, 1.0)):
    neurons = make_neurons(func_name, layer_input, neuron_count, func_weights_count, weight_bounds, func_bounds)
    result = _snn.Layer()
    result.add_neurons(neurons)
    return result


def rnd_weights(num, bounds=(-1.0, 1.0)):
    return bounds[0] + _rnd(num) * (bounds[1] - bounds[0])
