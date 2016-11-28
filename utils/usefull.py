import matplotlib.pyplot as _plt
import os as _os
import numpy as _np
import core.neuron_functions as _nf
import timeit as _timeit


def draw_neuron_plots(snn, dirname):
    _plt.figure()
    if not _os.path.exists(dirname):
        _os.mkdir(dirname)
    x = _np.linspace(-1, 1, 100)
    l = 1
    for layer in snn.layers():
        n = 1
        for neuron in layer:
            _plt.cla()
            f = _nf.get(neuron.func_name())
            if neuron.f_len() == 0:
                y = f(x)
            else:
                y = f(x, neuron.get_func_weights())
            _plt.plot(x, y, "b-")
            if neuron.f_len() == 0:
                _plt.title("Layer: %d, neuron: %d, func name: %s" % (
                           l, n, neuron.func_name()))
            else:
                _plt.title(
                    "Layer: %d, neuron: %d, func name: %s\nfunc_weight: %s" %
                    (l, n, neuron.func_name(), str(neuron.get_func_weights())))
            path = "layer_{}_neuron_{}.png".format(l, n)
            _plt.savefig(_os.path.join(dirname, path))
            n += 1
        l += 1


def benchmark_net(snn, times=5):
    w = snn.get_weights()
    t = _timeit.Timer(lambda: snn.error(w))
    seconds = t.timeit(times)
    return seconds