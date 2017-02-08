import logging as _log


class Teacher:
    """
        Base Teacher class. Teachers are used for "teaching"
        networks. How do they really teach depends on implementation.
        This base class have only logging_config field which
        setups logging settings
        Attributes:
              logging_config -- dict of kwargs which is sent directly
              to logging.baseConfig() function.
    """

    def __init__(self, logging_config=None):
        self._logging_config = logging_config
        if logging_config is not None and logging_config == "default":
            self._logging_config = dict()
            self._logging_config["format"] = "[%(asctime)s] %(levelname)s: %(message)s"
            self._logging_config["level"] = _log.NOTSET
        if self._logging_config is not None:
            _log.basicConfig(**self._logging_config)
        else:
            _log.basicConfig(level=_log.CRITICAL)
        self._log = _log.getLogger("teachers.{}".format(type(self).__name__))

    def teach(self, network, test_data, nb_epoch=None, eps=None,
              callback=None, separate_weights=False):
        """
            Main teach function. Call this function when you
            want to teach you net.
        :param network: instance of Perceptron class.
        :param test_data: dict in form {"x": inputs, "y": outputs} which
                          is used as train data.
        :param nb_epoch: Number of epochs. Optional.
        :param eps: epsilon. If current net error is < than eps teaching is complete. Optional.
        :param callback: callback(network, error) function which will be called when
                         one iteration of teaching is complete.
                         If such function returns False then teaching immediately
                         stopped and current net instance is returned.
        :param separate_weights: If True then first of all input weights are trained.
                                 Then functional weights are trained.
        :return: ready net, error
        """
        raise NotImplementedError()
