import snn.optimizers as _opt


class OptManager:

    def __init__(self, **kwargs):
        self.__kwargs = kwargs
        self.__methods = dict()

    def add_opt(self, name, **kwargs):
        self.__methods[name] = kwargs

    def __iter__(self):
        if not self.__methods:
            for name, className in _opt.get_all_methods().items():
                yield className(**self.__kwargs)
        else:
            for name, kwargs in self.__methods.items():
                yield _opt.get_method_class(name)(**{**self.__kwargs, **kwargs})
