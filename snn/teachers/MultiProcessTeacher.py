import logging
import multiprocessing as mp
import os.path as path
import tempfile
import time

import numpy as np

import snn.optimizers as optimizers
import snn


class MultiProcessTeacher:

    def __init__(self, optimizers,
                 network,
                 test_data,
                 process_num=-1,
                 logging_config=None):
        """

        :param optimizers: dictionary of optimizers which this teacher will use
         for minimizing an error function. Format: {"name:" <name-of-optimizer>,
                                                    "params": {<params>}}
        :param network: Instance of SNN object which should be taught.
        :param test_data: Test inputs for net. Format: {"x": <inputs>, "y": <outputs>
        :param process_num: Number of process. If not set defaults to cpu_count()
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
        self.__task["optimizers"] = optimizers
        self.__task["net_json"] = network.to_json(with_weights=False)
        self.__task["test_data"] = test_data
        if logging_config is not None:
            if logging_config == "default":
                self.__task["logging_config"] = {"format": "[%(asctime)s] %(levelname)s: %(message)s", "level": logging.NOTSET}
            else:
                self.__task["logging_config"] = logging_config
        self.__log.info("There will be %d processes.", self.__process_num)

    def teach(self, nb_epoch=None, eps=None):
        if nb_epoch is None and eps is None:
            return self.__teach()

        epochs_ended = False
        epochs_count = 0
        eps_reached = False
        best_result = None
        while not epochs_ended and not eps_reached:
            res = self.__teach()
            if best_result is None or res["error"] < best_result["error"]:
                best_result = res
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
        return best_result

    def __teach(self):
        start_time = time.perf_counter()
        self.__log.info("Starting teaching...")
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
    for opt in task["optimizers"]:
        log.info("Running \"%s\" with %s parameters.",
                 opt["name"], opt.get("params", {}))
        optimizer = optimizers.get_method_class(opt["name"])(opt.get("params", {}))
        weights = optimizer.start(network.error, weights)
        log.info("Optimizer \"%s\" has completed.", opt["name"])

    log.info("All %d optimizers have completed.",
             len(task["optimizers"]))
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
