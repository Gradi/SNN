from numpy import abs, array
from optimizers.base_optimizer import BaseOptimizer


class CoordinateDescent(BaseOptimizer):
    """
        Simple coordinate descent method.
        Attributes:
            h -- step in one direction. (Default: 1.0)
            h_mul -- multiplier of step(new h = h * h_mul) (Default: 0.7)
            eps -- epsilon (Default: 1e-3)
    """

    def __init__(self, params={}):
        super().__init__(params)
        self.__h = params.get("h", 1.0)
        self.__h_mul = params.get("h_mul", 0.7)
        self.__eps = params.get("eps", 1e-3)

    def start(self, f, x):
        iterations = 0
        ph = None
        h = self.__h
        iteration_successful = False

        while iterations < self._maxIter and \
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
                self._log.info("[Coordinate Descent] Progress: %5.0f%%."
                               " Iteration successful.",
                               iterations / self._maxIter * 100)
            else:
                self._log.info("[Coordinate Descent] Progress: %5.0f%%."
                               " Bad Iteration. Step progress: %5.0f%%.",
                               iterations / self._maxIter * 100,
                               self.__eps / (abs(ph - h)) * 100)
        return x

    def __minimize_direction(self, f, x, h):
        next_x = x + h
        while f(next_x) < f(x) and abs(f(x) - f(next_x)) > self.__eps:
            x = next_x
            next_x += h
            #self._log.info("[Coordinate Descent] X=%f, error=%f", x, f(x))
        return x


_optimizer_name = "coordinate_descent"
_class_type = CoordinateDescent
