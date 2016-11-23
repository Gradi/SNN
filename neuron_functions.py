import numpy as _np
import re as _re

"""
 All neuron function names must
 start with _neuron_func_<function-name>
 to allow functions to be added to _functions
 dictionary when this module is imported.
"""


def _neuron_func_linear(x):
    return x


def _neuron_func_quadratic(x):
    return _np.square(x)


def _neuron_func_cubic(x):
    return _np.power(x, 3)


def _neuron_func_ququadratic(x):
    return _np.power(x, 4)


def _neuron_func_simple_sigmoid(x, k=1):
    return (k * x) / (1 + _np.abs(k * x))

_functions = {}


def get(func_name):
    return _functions[func_name]


l = dict(locals())
for key, value in l.items():
    if _re.match("_neuron_func_", key):
        name = _re.split("_neuron_func_", key)[1]
        func = value
        _functions[name] = func