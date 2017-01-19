import logging


class BaseOptimizer:
    """
        Base optimizer class.
        This class needed only for setting up
        all general parameters of all optimizers and
        for setting up logger.
        Attributes:
            maxIter -- maximum number of iterations. (Default: 5000)

        Currently there is no any other general parameters.
    """

    def __init__(self, **kwargs):
        self._maxIter = kwargs.get("maxIter", 5000)
        self._log = logging.getLogger("optimizers.base_optimizer")
        self._log.disabled = True

    def start(self, f, x):
        msg = "This optimizer isn't implemented: {}".format(self)
        self._log.critical(msg)
        raise NameError(msg)
