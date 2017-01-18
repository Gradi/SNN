

def make_network(network_type, impl_type, *args, **kwargs):
    """
        Simple factory method that returns a class which implements
        a neural network.
    :param network_type: Type of neural network. Currently only "perceptron" is
                         supported.
    :param impl_type: Implementation type. Currently only "numpy" is supported.
    :param args: arguments for class.
    :param kwargs: kwarguments for class.
    """
    return _net_types[network_type](impl_type, *args, **kwargs)


def _perceptron_creator(impl_type, *args, **kwargs):
    raise NotImplementedError()

_net_types = dict()
_net_types["perceptron"] = _perceptron_creator