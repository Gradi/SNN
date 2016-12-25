import os
import importlib
import re

_path = locals()["__path__"][0]
_methods = dict()
_excluded = ["__init__.py", "base_optimizer.py", "optimizer_manager.py"]
for file in os.listdir(_path):
    fullname = os.path.join(_path, file)
    if os.path.isfile(fullname) and \
       re.search(r".*\.py", file) and \
       file not in _excluded:
        name = file[:-3]
        module = importlib.import_module("snn.optimizers.{}".format(name))
        _methods[module._optimizer_name] = module._class_type
del _path
del _excluded


def get_method_class(name):
    return _methods[name]


def get_all_methods():
    return _methods
