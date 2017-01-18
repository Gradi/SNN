

class Perceptron(object):

    def __init__(self, input_count, error_name):
        self._input_count = input_count
        self._error_name = error_name

    def set_test_inputs(self, inputs, out):
        raise NotImplementedError()

    def error(self, weights=None):
        raise NotImplementedError()

    def error_input_weights(self, input_weights=None):
        raise NotImplementedError()

    def error_func_weights(self, func_weights=None):
        raise NotImplementedError()

    def input(self, data):
        raise NotImplementedError()

    def set_weights(self, weights, weights_type=None):
        raise NotImplementedError()

    def get_weights(self, weights_type=None):
        raise NotImplementedError()

    def add_layer(self, layer):
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError()

    def net_output_len(self):
        raise NotImplementedError()

    def reset_weights(self, force=True):
        raise NotImplementedError()

    def to_json(self, with_weights=True):
        raise NotImplementedError()

    def save_to_file(self, filename, with_weights=True, overwrite=True):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def load_from_json(json_str):
        raise NotImplementedError()

    def load_from_file(filename):
        raise NotImplementedError()
