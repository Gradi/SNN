from numpy import abs, array
from snn.optimizers.base_optimizer import BaseOptimizer


class CoordinateDescent(BaseOptimizer):
    """
        Simple coordinate descent method.
        Attributes:
            h -- step in one direction. (Default: 1.0)
            h_good -- When we stepped in a good direction
                      new step = old step * h_good. (Default: 1.01).
            h_bad -- When we stepped in a bad direction
                      new step = old step * h_bad (Default: 0.7).
            eps -- epsilon (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__h      = kwargs.get("h", 1.0)
        self.__h_good = kwargs.get("h_good", 1.01)
        self.__h_bad  = kwargs.get("h_bad", 0.7)
        self.__eps    = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        iterations = 0
        h = self.__h
        ph = h + self.__eps * 5
        iteration_successful = False

        while iterations < self._maxIter and\
              (iteration_successful or abs(ph - h) > self.__eps):
            ph = h
            iteration_successful = False
            for i in range(0, len(x)):
                tx = array(x)

                def f_i(xi):
                    tx[i] = xi
                    return f(tx)

                tx[i] += h
                if f(tx) < f(x):
                    iteration_successful = True
                    x[i] = self.__minimize_direction(f_i, tx[i], h)
                else:
                    tx[i] -= 2 * h
                    h *= -1
                    if f(tx) < f(x):
                        iteration_successful = True
                        x[i] = self.__minimize_direction(f_i, tx[i], h)
            if iteration_successful:
                h *= self.__h_good
            else:
                h *= self.__h_bad
            iterations += 1

            if iteration_successful:
                self._log.info("[Coordinate Descent] Progress: %5.0f%%. "
                               "Iteration successful.",
                               iterations / self._maxIter * 100)
            else:
                self._log.info("[Coordinate Descent] Progress: %5.0f%%. "
                               "Bad Iteration. Step progress: %5.0f%%.",
                               iterations / self._maxIter * 100,
                               self.__eps / (abs(ph - h)) * 100)
        return x

    def __minimize_direction(self, f, x, h):
        max_h = h
        ph = 0

        f_curr = f(x)
        f_next = f(x + max_h)
        while f_next < f_curr and abs(f_curr - f_next) > (self.__eps / 100):
            f_curr = f_next
            ph = max_h
            max_h *= self.__h_good
            f_next = f(x + max_h)

        x += ph
        f_curr = f(x)
        f_next = f(x + h)
        while f_next < f_curr and abs(f_curr - f_next) > (self.__eps / 100):
            f_curr = f_next
            x += h
            f_curr = f(x)
        return x


_optimizer_name = "coordinate_descent"
_class_type = CoordinateDescent
