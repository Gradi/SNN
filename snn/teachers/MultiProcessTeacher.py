import multiprocessing as _mp
import numpy as _np
import os.path as _path
import logging as _log
import tempfile, time


from snn.optimizers.optimizer_manager import OptManager
from snn.teachers.teacher import Teacher
import snn


class MultiProcessTeacher(Teacher):

    def __init__(self, opt_manager, process_num=-1, logging_config=None):
        """
        :param opt_manager: Instance of OptManager class.
        :param process_num: Number of processes. If not set defaults to cpu_count().
        :param logging_config: See base class Teacher.
        """
        super().__init__(logging_config)
        if process_num == -1:
            self.__process_num = _mp.cpu_count()
        else:
            self.__process_num = process_num
        self.__result_queue = _mp.Queue()
        self.__opt_manager = opt_manager
        self.__net_json   = None
        self.__test_data  = None
        self.__result_dir = None
        self._log.info("There will be %d processes.", self.__process_num)

    def teach(self, network, test_data, nb_epoch=None, eps=None,
              callback=None):
        network = network.copy()

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

            self._log.info("Current error: %f, best error: %f", res["error"],
                           best_result["error"])
            if nb_epoch is not None:
                epochs_count += 1
                self._log.info("Epoch: %d/%d (%3.2f%%)", epochs_count, nb_epoch,
                               epochs_count / nb_epoch * 100)
                if epochs_count >= nb_epoch:
                    epochs_ended = True
            if eps is not None:
                eps_reached = best_result["error"] <= eps
                self._log.info("Epsilon: %.2f (%10.5f%%)", best_result["error"] / eps,
                               eps / best_result["error"] * 100)
            if callback is not None:
                network.set_weights(best_result["weights"])
                if not callback(network, best_result["error"]):
                    break

        network.set_weights(best_result["weights"])
        return network, best_result["error"]

    def __teach(self, network, test_data):
        start_time = time.time()
        self._log.info("Starting teaching...")
        self.__net_json = network.to_json(with_weights=False)
        self.__test_data = test_data
        result_dir = tempfile.TemporaryDirectory()
        self.__result_dir = result_dir.name
        self._log.info("Temporary dir is: %s", self.__result_dir)
        workers = list()
        for i in range(0, self.__process_num):
            process = _mp.Process(target=self._worker_main,
                                  name="Teacher #%d" % i)
            process.daemon = True
            workers.append(process)
            process.start()
        for worker in workers:
            worker.join()
        del workers
        self._log.info("Teaching is complete.")

        self._log.info("Looking for best network.")
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
        self._log.info("Cleaning up a temporary directory.")
        result_dir.cleanup()
        self.__net_json   = None
        self.__test_data  = None
        self.__result_dir = None
        sec = time.time() - start_time
        self._log.info("Teaching took %.5f seconds(%.2f min).", sec, sec / 60)
        return best_result

    def _worker_main(self):
        if self.__logging_config is not None:
            logging_config = self.__logging_config
            logging_config["format"] = "[{}] {}".format(_mp.current_process().name,
                                                        logging_config["format"])
            _log.basicConfig(**logging_config)
        else:
            _log.disable(_log.CRITICAL)
        # Log instance can't be pickled that is why we need to create a new log.
        log = _log.getLogger("teachers._worker_main")

        network = snn.load_from_json(self.__net_json)
        network.set_test_inputs(self.__test_data["x"], self.__test_data["y"])

        weights = network.get_weights()
        for opt in self.__opt_manager:
            log.info("Running \"%s\".", type(opt).__name__)
            weights = opt.start(network.error, weights)
            network.set_weights(weights)
            log.info("Optimizer \"%s\" has completed.", type(opt).__name__)

        log.info("Ran all the optimizers.")
        filename = _path.join(self.__result_dir, "tmp_weights_{}.npz".
                              format(_mp.current_process().pid))
        _np.savez_compressed(filename, weights=network.get_weights())
        log.info("Saved current weights to \"%s\"", filename)
        result = dict()
        result["error"] = network.error()
        result["filename"] = filename
        self.__result_queue.put(result)
        self.__result_queue.close()
        log.info("Done.")
