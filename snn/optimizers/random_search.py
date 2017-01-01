from numpy import abs
from numpy.random import rand
from snn.optimizers.base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
        Random search method.
        Attributes:
              rad -- radius (Default: 1)
              step_increase -- How much increase step when method takes good
                               direction. (Default: 1.5)
              step_decrease -- How much decrease step when method can't take
                               any good direction. (Default: 0.6)
              max_attempts -- How much method should make attempts before step_decrease.
                          (Default: 1000)
              eps -- Epsilon. (Default: 1e-3)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__rad = kwargs.get("rad", 1)
        self.__step_increase = kwargs.get("step_increase", 1.5)
        self.__step_decrease = kwargs.get("step_decrease", 0.6)
        self.__max_attempts = kwargs.get("max_attempts", 1000)
        self.__eps = kwargs.get("eps", 1e-3)

    def start(self, f, x):
        rad = self.__rad
        prev_rad = None
        iterations = 0

        while iterations < self._maxIter and \
                (prev_rad is None or abs(prev_rad - rad) > self.__eps):
            prev_rad = rad
            next_point = x + (-rad + rand(x.size) * rad * 2)
            fx = f(x)
            attempts = 0
            while f(next_point) > fx and attempts < self.__max_attempts:
                next_point = x + (-rad + rand(x.size) * rad * 2)
                attempts += 1
            if attempts < self.__max_attempts:
                rad *= self.__step_increase
                x = next_point
            else:
                rad *= self.__step_decrease
            iterations += 1
            self._log.info("[Random Search] Iteration progress: %3.2f%%,"
                           " Attempts this time: %d/%d (%3.2f%%)" % (
                            iterations / self._maxIter * 100,
                            attempts, self.__max_attempts,
                            attempts / self.__max_attempts * 100))

        return x

_optimizer_name = "random_search"
_class_type = RandomSearch
