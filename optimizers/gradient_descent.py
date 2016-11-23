import numpy as _np
from optimizers.base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """
        Simple gradient descent method.
        Attributes:
            h -- step (Default: 1).
            h_mul -- value which is multiplier for step(new_h = h * h_mul) (Default: 0.5).
    """

    def __init__(self, params={}):
        super().__init__(params)
        self.__h = params.get("h", 1)
        self.__h_mul = params.get("h_mul", 0.5)

    def start(self, f, x, check_bounds=None):
        x = _np.array(x)
        h = self.__h

        iters = 0
        while iters < self._maxIter:
            gr_x = self.__grad(f, x)
            new_x = x + -1 * gr_x * h
            if check_bounds is not None:
                check_bounds(new_x)
            if f(x) < f(new_x):
                h *= self.__h_mul
            else:
                x = new_x
            iters += 1
            self._log.info("[Gradient Descent] Progress: %3.0f%%",
                           iters / self._maxIter * 100)
        return x

    def __grad(self, f, x, dx=1e-5, norm=True):
        if not hasattr(x, "__iter__"):
            return (f(x + dx) - f(x)) / dx
        else:
            y = list()
            for i in range(0, len(x)):
                new_x = _np.array(x)
                new_x[i] += dx
                y.append((f(new_x) - f(x)) / dx)
            y = _np.array(y)
            if norm:
                length = _np.sqrt(_np.sum(y ** 2))
                y /= length
            return y


_optimizer_name = "gradient_descent"
_class_type = GradientDescent
