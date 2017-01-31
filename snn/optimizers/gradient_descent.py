import numpy as _np
from snn.optimizers.base_optimizer import BaseOptimizer
from snn.optimizers.utils import gradient as _grad


class GradientDescent(BaseOptimizer):
    """
        Simple gradient descent method.
        Attributes:
            h -- step (Default: 1).
            h_good -- When we stepped in a good direction
                      new step = old step * h_good. (Default: 1.01).
            h_bad -- When we stepped in a bad direction
                      new step = old step * h_bad (Default: 0.7).
            eps: (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__h      = kwargs.get("h", 1)
        self.__h_good = kwargs.get("h_good", 1.01)
        self.__h_bad  = kwargs.get("h_bad", 0.7)
        self.__eps    = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        x = _np.array(x)
        h = self.__h
        ph = h + self.__eps * 5

        iters = 0
        good_iteration = True
        while iters < self._maxIter and\
              (good_iteration or abs(ph - h) > self.__eps):
            ph = h
            gr_x = _grad(f, x)
            new_x = x + -1.0 * gr_x * h
            if f(x) < f(new_x):
                h *= self.__h_bad
                good_iteration = False
            else:
                x = new_x
                h *= self.__h_good
                good_iteration = True
            iters += 1
            self._log.info("[Gradient Descent] Progress: %3.0f%%",
                           iters / self._maxIter * 100)
        return x


_optimizer_name = "gradient_descent"
_class_type = GradientDescent
