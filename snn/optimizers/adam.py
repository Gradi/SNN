import numpy as _np
from snn.optimizers.base_optimizer import BaseOptimizer
from snn.optimizers.utils import gradient as _grad


class Adam(BaseOptimizer):
    """
        This is implementation of Adam optimization method.
        For details see https://arxiv.org/abs/1412.6980
        P.S. This method ignores maxIter attribute.
        Attributes:
              h -- stepsize. (Default: 0.001)
              B1 -- first moment. (Default: 0.9)
              B2 -- second moment. (Default: 0.999)
              e -- that tricky constant that makes method more stable? (Default: 1e-8)
              eps -- Method stops when abs(previous_value - current_value) <= eps. (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__h   = kwargs.get("h", 0.001)
        self.__B1  = kwargs.get("B1", 0.9)
        self.__B2  = kwargs.get("B2", 0.999)
        self.__e   = kwargs.get("e", 1e-8)
        self.__eps = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        pm = _np.zeros(x.shape, dtype=_np.float) # p means previous.
        pv = _np.zeros(x.shape, dtype=_np.float)
        t = 0
        minimum_x = _np.array(x)
        minimum_f = f(x)
        pf = minimum_f + self.__eps * 100

        while _np.abs(pf - f(x)) > self.__eps:
            t += 1
            g = _grad(f, x)
            m = self.__B1 * pm + (1.0 - self.__B1) * g
            v = self.__B2 * pv + (1.0 - self.__B2) * _np.square(g)
            pm = m
            pv = v
            m = m / (1.0 - _np.power(self.__B1, t))
            v = v / (1.0 - _np.power(self.__B2, t))
            pf = f(x)
            h = self.__h * _np.sqrt(1.0 - _np.power(self.__B2, t)) \
                / (1.0 - _np.power(self.__B1, t))
            x = x - h * m / (_np.sqrt(v) + self.__e)
            current_f = f(x)
            # I think it is good idea to remember
            # the best minimum we ever found.
            # Thanks to this if method takes bad direction
            # we, at least, get some local minimum.
            if current_f < minimum_f:
                minimum_f = current_f
                minimum_x = _np.array(x)

        return minimum_x


_optimizer_name = "adam"
_class_type = Adam