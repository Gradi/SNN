import time
import numpy as _np

from snn.optimizers.optimizer_manager import OptManager
from snn.teachers.teacher import Teacher
import snn


class SimpleTeacher(Teacher):
    """
        Simple implementation of Teacher class.
        Accepts network and run all optimizers from opt_manager instance on it.
    """

    def __init__(self, opt_manager, logging_config=None):
        super().__init__(logging_config)
        self.__opt_manager = opt_manager

    def teach(self, network, test_data, nb_epoch=None, eps=None,
              callback=None, separate_weights=False):
        start_time = time.time()

        network = network.copy()

        epochs_ended = False
        eps_reached = False
        epochs_count = 0
        network.set_test_inputs(test_data["x"], test_data["y"])
        best_weights = None
        best_error = None

        while not epochs_ended and not eps_reached:
            network.reset_weights()
            if not separate_weights:
                for opt in self.__opt_manager:
                    self._log.info("Running \"%s\"", type(opt).__name__)
                    weights = opt.start(network.error, network.get_weights())
                    network.set_weights(weights)
            else:
                error = 0
                prev_error = 1
                prev_net = None
                while error <= prev_error:
                    prev_error = error
                    prev_net = network.copy()
                    weights = network.get_weights("input")
                    func_weights = network.get_weights("func")

                    self._log.info("Training input weights only.")
                    for opt in self.__opt_manager:
                        self._log.info("Running \"%s\"", type(opt).__name__)
                        weights = opt.start(network.error_input_weights,
                                            weights)
                        network.set_weights(weights, "input")
                        self._log.info("Optimizer \"%s\" has completed.",
                                 type(opt).__name__)

                    self._log.info("Training func weights only.")
                    for opt in self.__opt_manager:
                        self._log.info("Running \"%s\"", type(opt).__name__)
                        func_weights = opt.start(network.error_func_weights,
                                                 func_weights)
                        network.set_weights(func_weights, "func")
                        self._log.info("Optimizer \"%s\" has completed.",
                                 type(opt).__name__)

                    log.info("Training all weights together.")
                    for opt in self.__opt_manager:
                        log.info("Running \"%s\"", type(opt).__name__)
                        weights = opt.start(network.error, network.get_weights())
                        network.set_weights(weights)
                        log.info("Optimizer \"%s\" has completed.", type(opt).__name__)

                    error = network.error()
                network = prev_net

            self._log.info("Ran all the optimizers.")
            if best_error is None or network.error() < best_error:
                best_error = network.error()
                best_weights = network.get_weights()

            if nb_epoch is not None:
                epochs_count += 1
                epochs_ended = epochs_count >= nb_epoch
                self._log.info("Epochs: %d/%d (%3.2f%%)", epochs_count, nb_epoch,
                               epochs_count / nb_epoch * 100)
            if eps is not None:
                eps_reached = best_error < eps
                self._log.info("Epsilon: %5.5f%%", eps / best_error * 100)
            if eps is None and nb_epoch is None:
                break
            if callback is not None:
                network.set_weights(best_weights)
                if not callback(network, best_error):
                    break

        network.set_weights(best_weights)
        sec = time.time() - start_time
        self._log.info("Teaching is complete. Took time: %5.3f sec (%3.2f min)",
                       sec, sec / 60)
        return network, best_error
