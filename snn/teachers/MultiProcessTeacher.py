import logging
import multiprocessing as mp
import os.path as path
import tempfile
import time

import numpy as np

from snn.optimizers.optimizer_manager import OptManager
import snn


class MultiProcessTeacher:

    def __init__(self,
                 opt_manager,
                 process_num=-1,
                 logging_config=None):
        """

        :param opt_manager: Instance of OptManager class.
        :param network: Instance of SNN object which should be taught.
        :param process_num: Number of process. If not set defaults to cpu_count().
        :param logging_config: Logging config for loggind.basicConfig() function.
                If not set then no logs will be produced by child processes.
                If set to "default" then default log config will be used.
        """
        self.__log = logging.getLogger("teachers.MultiProcessTeacher")
        if process_num == -1:
            self.__process_num = mp.cpu_count()
        else:
            self.__process_num = process_num
        self.__result_queue = mp.Queue()
        self.__task = dict()
        self.__task["opt_manager"] = opt_manager
        if logging_config is not None:
            if logging_config == "default":
                self.__task["logging_config"] = {"format": "[%(asctime)s] %(levelname)s: %(message)s", "level": logging.NOTSET}
            else:
                self.__task["logging_config"] = logging_config
        self.__log.info("There will be %d processes.", self.__process_num)

    def teach(self, network, test_data, nb_epoch=None, eps=None,
              backup_filename=None):
        network = network.copy()
        if nb_epoch is None and eps is None:
            res = self.__teach(network, test_data)
            network.set_weights(res["weights"])
            return network, res["error"]

        epochs_ended = False
        epochs_count = 0
        eps_reached = False
        best_result = None

        while not epochs_ended and not eps_reached:
            res = self.__teach(network, test_data)
            if best_result is None or res["error"] < best_result["error"]:
                best_result = res
                if backup_filename is not None:
                    network.set_weights(best_result["weights"])
                    network.save_to_file(backup_filename)

            self.__log.info("Current error: %f, best error: %f", res["error"],
                            best_result["error"])
            if nb_epoch is not None:
                epochs_count += 1
                self.__log.info("Epoch: %d/%d (%3.2f%%)", epochs_count, nb_epoch,
                                epochs_count / nb_epoch * 100)
                if epochs_count >= nb_epoch:
                    epochs_ended = True
            if eps is not None:
                eps_reached = best_result["error"] <= eps
        network.set_weights(best_result["weights"])
        return network, best_result["error"]

    def __teach(self, network, test_data):
        start_time = time.perf_counter()
        self.__log.info("Starting teaching...")
        self.__task["net_json"] = network.to_json(False)
        self.__task["test_data"] = test_data
        result_dir = tempfile.TemporaryDirectory()
        self.__log.info("Temporary dir is: %s", result_dir.name)
        self.__task["result_dir"] = result_dir.name
        workers = list()
        for i in range(0, self.__process_num):
            process = mp.Process(target=_worker_main,
                                 args=(self.__task, self.__result_queue),
                                 name="Teacher #%d" % i)
            process.daemon = True
            workers.append(process)
            process.start()
        for worker in workers:
            worker.join()
        del workers
        self.__log.info("Teaching is complete.")

        self.__log.info("Looking for best network.")
        best_result = None
        while not self.__result_queue.empty():
            res = self.__result_queue.get()
            if best_result is None or res["error"] < best_result["error"]:
                arrs = np.load(res["filename"])
                weights = arrs["weights"]
                arrs.close()
                best_result = dict()
                best_result["error"] = res["error"]
                best_result["weights"] = weights
        self.__log.info("Cleaning up a temporary directory.")
        result_dir.cleanup()
        sec = time.perf_counter() - start_time
        self.__log.info("Teaching took %f seconds(%.0f min).", sec, sec / 60)
        return best_result


def _worker_main(task, result_queue):
    log = logging.getLogger("teachers._worker_main")
    if "logging_config" in task:
        logging_config = task["logging_config"]
        logging_config["format"] = "[{}] {}".format(mp.current_process().name,
                                                    logging_config["format"])
        logging.basicConfig(**logging_config)
    else:
        logging.disable(logging.CRITICAL)

    result_dir = task["result_dir"]

    network = snn.load_from_json(task["net_json"])
    network.set_test_inputs(task["test_data"]["x"], task["test_data"]["y"])
    weights = network.get_weights()
    for opt in task["opt_manager"]:
        log.info("Running \"%s\".", type(opt).__name__)
        weights = opt.start(network.error, weights)
        log.info("Optimizer \"%s\" has completed.", type(opt).__name__)

    log.info("Ran all the optimizers.")
    filename = path.join(result_dir, "tmp_weights_{}.npz"
                         .format(mp.current_process().pid))
    np.savez_compressed(filename, weights=weights)
    log.info("Saved current weights to \"%s\"", filename)
    result = dict()
    result["error"] = network.error()
    result["filename"] = filename
    result_queue.put(result)
    result_queue.close()
    log.info("Done.")
