from numpy.random import rand as _rnd
import numpy as _np


class DataGen:
    def __init__(self, f, input_count, input_bounds, error=0):
        if len(input_bounds) > input_count:
            raise NameError("len of bounds > input count")
        self.f = f
        self.input_count = input_count
        self.error = error / 100.0
        self.bounds = _np.array(input_bounds)
        if len(self.bounds) < self.input_count:
            for i in (range(0, self.input_count - len(self.bounds))):
                self.bounds = _np.append(self.bounds, [input_bounds[-1]], axis=0)

    def next(self, count=1):
        if count == 1:
            return self.__next()
        else:
            x = []
            y = []
            for i in range(0, count):
                new_x, new_y = self.__next()
                x.append(new_x)
                y.append(new_y)
            x = _np.array(x)
            y = _np.array(y)
            return x, y

    def __next(self):
        x = []
        for i in range(0, self.input_count):
            x.append(self.bounds[i][0] + _rnd() * (self.bounds[i][1] - self.bounds[i][0]))
        x = _np.array(x)
        y = self.f(x)
        if not isinstance(y, _np.ndarray):
            y = _np.array(y)
        if self.error != 0:
            y *= self.__rnd_error()
        return x, y

    def __rnd_error(self):
        return (1.0 - self.error) + _rnd() * 2 * self.error
