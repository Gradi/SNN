import numpy as _np


def gradient(f, x, dx=1e-3, norm=True):
    """
    :param f: target function.
    :param x: vector x
    :param dx: step size. Default: 1e-3
    :param norm: Normalize vector.
    :return: gradient vector.
    """

    old_shape = x.shape
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    result = list()
    for x in x:
        tmp_res = _np.empty(len(x), dtype=_np.float)
        for i in range(0, len(x)):
            tmp_x = _np.array(x)
            tmp_x[i] += dx
            f1 = f(tmp_x)
            tmp_x[i] -= 2 * dx
            f2 = f(tmp_x)
            dy = (f1 - f2) / (2 * dx)
            tmp_res[i] = dy
        result.append(tmp_res)
    result = _np.array(result).reshape(old_shape)
    if norm:
        length = _np.sum(_np.square(result))
        length = _np.sqrt(length)
        if length:
            result /= length
    return _np.array(result).reshape(old_shape)