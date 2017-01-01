import logging
import multiprocessing as mp
import os.path as path
import tempfile
import time

import numpy as _np

from snn.optimizers.optimizer_manager import OptManager
import snn


class MultiProcessTeacher:

    def __init__(self,
                 opt_manager,
                 process_num=-1,
                 logging_config=None):
        """

        :param opt_manager: Instance of OptManager class.
        :param process_num: Number of process. If not set defaults to cpu_count().
        :param logging_config: Logging config for loggind.basicConfig() function.
                If not set then no logs will be produced by child processes.
                If set to "default" then default log config will be used.
        """
        if process_num == -1:
            self.__process_num = mp.cpu_count()
        else:
            self.__process_num = process_num
        self.__result_queue = mp.Queue()
        self.__opt_manager = opt_manager
        if logging_config is not None:
            if logging_config == "default":
                self.__logging_config = {"format": "[%(asctime)s] %(levelname)s: %(message)s", "level": logging.NOTSET}
            else:
                self.__logging_config = logging_config
        else:
            self.__logging_config = None
        self.__separate_weights = False
        self.__eps = 0.0

    def teach(self, network, test_data, separate_weights=False,
              nb_epoch=None, eps=None):
        log = logging.getLogger("teachers.MultiProcessTeacher")
        network = network.copy()
        self.__separate_weights = separate_weights
        self.__eps = eps
        if separate_weights and eps is None:
            raise NameError("You should set eps when separate_weights is True.")

        if nb_epoch is None and eps is None:
            res = self.__teach(network, test_data)
            network.set_weights(res["weights"])
            return network, res["error"]

        epochs_ended = False
        eps_reached = False
        epochs_count = 0
        best_result = None

        while not epochs_ended and not eps_reached:
            res = self.__teach(network, test_data)
            if best_result is None or res["error"] < best_result["error"]:
                best_result = res

            log.info("Current error: %f, best error: %f", res["error"],
                            best_result["error"])
            if nb_epoch is not None:
                epochs_count += 1
                log.info("Epoch: %d/%d (%3.2f%%)", epochs_count, nb_epoch,
                                epochs_count / nb_epoch * 100)
                if epochs_count >= nb_epoch:
                    epochs_ended = True
            if eps is not None:
                eps_reached = best_result["error"] <= eps
        network.set_weights(best_result["weights"])
        return network, best_result["error"]

    def __teach(self, network, test_data):
        start_time = time.time()
        log = logging.getLogger("teachers.MultiProcessTeacher")
        log.info("Starting teaching...")
        self.__net_json = network.to_json(False)
        self.__test_data = test_data
        result_dir = tempfile.TemporaryDirectory()
        self.__result_dir = result_dir.name
        log.info("Temporary dir is: %s", result_dir.name)
        workers = list()
        for i in range(0, self.__process_num):
            process = mp.Process(target=self._worker_main,
                                 name="Teacher #%d" % i)
            process.daemon = True
            workers.append(process)
            process.start()
        for worker in workers:
            worker.join()
        del workers
        log.info("Teaching is complete.")

        log.info("Looking for best network.")
        best_result = None
        while not self.__result_queue.empty():
            res = self.__result_queue.get()
            if best_result is None or res["error"] < best_result["error"]:
                arrs = _np.load(res["filename"])
                weights = arrs["weights"]
                arrs.close()
                best_result = dict()
                best_result["error"] = res["error"]
                best_result["weights"] = weights
        log.info("Cleaning up a temporary directory.")
        result_dir.cleanup()
        sec = time.time() - start_time
        log.info("Teaching took %.5f seconds(%.2f min).", sec, sec / 60)
        return best_result

    def _worker_main(self):
        if self.__logging_config is not None:
            logging_config = self.__logging_config
            logging_config["format"] = "[{}] {}".format(mp.current_process().name,
                                                        logging_config["format"])
            logging.basicConfig(**logging_config)
        else:
            logging.disable(logging.CRITICAL)
        log = logging.getLogger("teachers._worker_main")

        network = snn.load_from_json(self.__net_json)
        network.set_test_inputs(self.__test_data["x"], self.__test_data["y"])

        if not self.__separate_weights:
            weights = network.get_weights()
            for opt in self.__opt_manager:
                log.info("Running \"%s\".", type(opt).__name__)
                weights = opt.start(network.error, weights)
                network.set_weights(weights)
                log.info("Optimizer \"%s\" has completed.", type(opt).__name__)
        else:
            input_weights = network.get_weights("input")
            func_weights = network.get_weights("func")
            error = network.error()
            prev_error = error + 2 * self.__eps
            while _np.abs(error - prev_error) > self.__eps:

                def f(func_weights):
                    network.set_weights(func_weights, "func")
                    input_weights = network.get_weights("input")
                    for opt in self.__opt_manager:
                        input_weights = opt.start(network.weights_only_error, input_weights)
                        network.set_weights(input_weights, "input")
                    return network.error()

                for opt in self.__opt_manager:
                    func_weights = opt.start(f, func_weights)
                    network.set_weights(func_weights, "func")

                prev_error = error
                error = network.error()

        log.info("Ran all the optimizers.")
        filename = path.join(self.__result_dir, "tmp_weights_{}.npz"
                             .format(mp.current_process().pid))
        _np.savez_compressed(filename, weights=network.get_weights())
        log.info("Saved current weights to \"%s\"", filename)
        result = dict()
        result["error"] = network.error()
        result["filename"] = filename
        self.__result_queue.put(result)
        self.__result_queue.close()
        log.info("Done.")
