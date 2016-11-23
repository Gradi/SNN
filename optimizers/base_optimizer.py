import logging


class BaseOptimizer:
    """
        Base optimizer class.
        This class needed only for setting up
        all general parameters of all optimizers and
        for setting up logger.
        Attributes:
            maxIter -- maximum number of iterations. (Default: 200)

        Currently there is no any other general parameters.
    """

    def __init__(self, params={}):
        self._maxIter = params.get("maxIter", 200)
        self._log = logging.getLogger("optimizers.base_optimizer")

    def start(self, f, x, check_bounds=None):
        msg = "This optimizer isn't implemented: {}".format(self)
        self._log.critical(msg)
        raise NameError(msg)
