import multiprocessing as mp
import tempfile
import logging
import numpy as np
import os.path as path
import time
import optimizers
import SNN


class MultiProcessTeacher:

    def __init__(self, optimizers,
                 net_json,
                 test_data,
                 process_num=-1,
                 logging_config=None):
        self.__log = logging.getLogger("teachers.MultiProcessTeacher")
        if process_num == -1:
            self.__process_num = mp.cpu_count()
        else:
            self.__process_num = process_num
        self.__result_queue = mp.Queue()
        self.__task = dict()
        self.__task["optimizers"] = optimizers
        self.__task["net_json"] = net_json
        self.__task["test_data"] = test_data
        if logging_config is not None:
            self.__task["logging_config"] = logging_config
        self.__log.info("There will be %d processes.", self.__process_num)

    def teach(self):
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
    snn = SNN.load_from_json(task["net_json"])
    network = SNN.OptSNN(snn)
    network.set_test_data(task["test_data"]["x"], task["test_data"]["y"])
    weights = network.get_weights()
    for opt in task["optimizers"]:
        log.info("Running \"%s\" with %s parameters.",
                 opt["name"], opt["params"])
        optimizer = optimizers.get_method_class(opt["name"])(opt.get("params", {}))
        weights = optimizer.start(network.error, weights)
        log.info("Optimizer \"%s\" has completed.", opt["name"])
    log.info("All %d optimizers have completed.",
             len(task["optimizers"]))
    filename = path.join(result_dir, "tmp_weights_{}.npz"
                         .format(mp.current_process().pid))
    good_weights = network.to_simple_snn().get_weights()
    np.savez_compressed(filename, weights=good_weights)
    log.info("Saved current weights to \"%s\"", filename)
    result = dict()
    result["error"] = network.error(weights)
    result["filename"] = filename
    result_queue.put(result)
    result_queue.close()
    log.info("Done.")
