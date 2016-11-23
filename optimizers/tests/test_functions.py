import numpy as _np

"""
    This file contains functions which are
    used for testing different methods of
    optimizations.
    _functions is a list which contains
    dictionaries.
    That dictionaries look like this:
    {
        "f": target function,
        "num": size of input vector of f,
        "min": good min point,
        "bounds": search bounds,
        "atol": Maximum difference between
        num minimum and good minimum.
    }
"""

_functions = list()

_a = 1.0
_b = 100.0
def _rosenbrock_function(x):
    x, y = x
    return (_a - x) ** 2 + _b * (y - x ** 2) ** 2
_functions.append({"f": _rosenbrock_function, "num": 2,
                   "min": _np.array([_a, _a ** 2]),
                   "bounds": (-2.0, 2.0), "atol": 0.4})


def _simple_function_001(x):
    x, y = x
    return 8 * x ** 2 + 5 * y ** 2 + 4 * x * y
_functions.append({"f": _simple_function_001, "num": 2,
                   "min": _np.array([0.0, 0.0]),
                   "bounds": (-4.0, 4.0), "atol": 1e-3})


def _check_bounds(x, bounds):
    for i in range(0, len(x)):
        x[i] = _np.maximum(x[i], bounds[0])
        x[i] = _np.minimum(x[i], bounds[1])


def _rnd_point(num, bounds):
    """
    Functions which is called from test
    classes to generate point.
    :param num: size of vector
    :param bounds: bounds in which point is generated
    :return:
    """
    return bounds[0] + _np.random.rand(num) * (bounds[1] - bounds[0])
