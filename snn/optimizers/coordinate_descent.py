from numpy import abs, array
from snn.optimizers.base_optimizer import BaseOptimizer


class CoordinateDescent(BaseOptimizer):
    """
        Simple coordinate descent method.
        Attributes:
            h -- step in one direction. (Default: 1.0)
            h_bad -- When we stepped in a bad direction
                      new step = old step * h_bad (Default: 0.9).
            eps -- epsilon (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__h      = kwargs.get("h", 1.0)
        self.__h_bad  = kwargs.get("h_bad", 0.9)
        self.__eps    = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        iterations = 0
        h = self.__h
        pf = f(x) + self.__eps * 5
        iteration_successful = False



        while iterations < self._maxIter and abs(pf - f(x)) > self.__eps:
            iteration_successful = False
            for i in range(0, len(x)):
                tx = array(x)

                def f_i(xi):
                    tx[i] = xi
                    return f(tx)

                tx[i] += h
                if f(tx) < f(x):
                    iteration_successful = True
                    pf = f(x)
                    x[i] = self.__minimize_direction(f_i, tx[i], h)
                else:
                    tx[i] -= 2 * h
                    h *= -1
                    if f(tx) < f(x):
                        iteration_successful = True
                        pf = f(x)
                        x[i] = self.__minimize_direction(f_i, tx[i], h)
            if not iteration_successful:
                h *= self.__h_bad

            iterations += 1
            # self._log.info("[Coordinate Descent]\n"
            #                "\tIteration: %3.2f%% (%s).\n"
            #                "\tCurrent h: %f\n"
            #                "\tFunc value: %f",
            #                 iterations / self._maxIter * 100,
            #                 "good" if iteration_successful else "bad",
            #                 h,
            #                 f(x))
        return x

    def __minimize_direction(self, f, x, h):
        max_h = h
        ph = 0

        f_curr = f(x)
        f_next = f(x + max_h)
        while f_next < f_curr and abs(f_curr - f_next) > (self.__eps / 1000.0):
            f_curr = f_next
            ph = max_h
            max_h *= 1.1
            f_next = f(x + max_h)

        x += ph
        while abs(h) > self.__eps:
            f_curr = f(x)
            f_next = f(x + h)
            while f_next < f_curr and abs(f_next - f_curr) > (self.__eps / 1000.0):
                f_curr = f_next
                x += h
                f_next = f(x)
            h *= self.__h_bad
        return x


_optimizer_name = "coordinate_descent"
_class_type = CoordinateDescent
