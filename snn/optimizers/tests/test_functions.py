import numpy as _np

"""
    This file contains functions which are
    used for testing different methods of
    optimizations.
    _functions is a list which contains
    dictionaries.
    Each dictionary looks like this:
    {
        "f": target function,
        "num": size of input vector of f,
        "min": good min point,
        "bounds": start point bounds,
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
_functions.append({"f": _rosenbrock_function,
                   "num": 2,
                   "min": _np.array([_a, _a ** 2]),
                   "bounds": (-7.0, -7.5),
                   "atol": 0.09})


def _simple_function_001(x):
    x, y = x
    return 8 * x ** 2 + 5 * y ** 2 + 4 * x * y
_functions.append({"f": _simple_function_001,
                   "num": 2,
                   "min": _np.array([0.0, 0.0]),
                   "bounds": (-10.0, 10.0),
                   "atol": 0.1})


def _sphere5d(x):
    sum = 0
    for x in x:
        sum += x ** 2
    return sum
_functions.append({"f": _sphere5d,
                   "num": 5,
                   "min": _np.zeros(5, dtype=_np.float),
                   "bounds": (-10, 10),
                   "atol": 0.1})


def _rnd_point(num, bounds):
    """
    Function which is called from test
    classes to generate point.
    :param num: size of vector
    :param bounds: bounds in which point is generated
    """
    return bounds[0] + _np.random.rand(num) * (bounds[1] - bounds[0])
