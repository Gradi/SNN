from numpy import abs, array
from snn.optimizers.base_optimizer import BaseOptimizer


class CoordinateDescent(BaseOptimizer):
    """
        Simple coordinate descent method.
        Attributes:
            h -- step in one direction. (Default: 1.0)
            h_mul -- multiplier of step(new_h = h * h_mul) (Default: 0.7)
            eps -- epsilon (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__h = kwargs.get("h", 1.0)
        self.__h_mul = kwargs.get("h_mul", 0.7)
        self.__eps = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        iterations = 0
        ph = None
        h = self.__h
        iteration_successful = False

        while iterations < self._maxIter and\
              (ph is None or iteration_successful or abs(ph - h) > self.__eps):
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
                    if f(tx) < f(x):
                        iteration_successful = True
                        x[i] = self.__minimize_direction(f_i, tx[i], -h)
            if not iteration_successful:
                h *= self.__h_mul
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
        fx = f(x)

        while f(x + max_h) < fx:
            ph = max_h
            max_h /= self.__h_mul

        x += ph
        fx = f(x)
        while f(x + h) < fx:
            x += h
        return x


_optimizer_name = "coordinate_descent"
_class_type = CoordinateDescent
