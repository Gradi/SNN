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
        self.__nb_epoch   = None
        self.__eps        = None
        self.__separate_weights = False
        self.__result_dir = None
        self._log.info("There will be %d processes.", self.__process_num)

    def teach(self, network, test_data, nb_epoch=None, eps=None,
              callback=None, separate_weights=False):
        self.__net_json = network.to_json(False)
        self.__test_data = test_data
        self.__nb_epoch = nb_epoch
        self.__eps = eps
        self.__separate_weights = separate_weights
        if callback is not None:
            self._log.warn("Currently callback function is not supported in "
                           "MultiProcessTeacher.")
        weights, error = self._get_best_weights_error()
        network = network.copy()
        network.set_weights(weights)
        return network, error

    def _get_best_weights_error(self):
        start_time = time.time()
        self._log.info("Starting teaching...")
        result_dir = tempfile.TemporaryDirectory()
        self.__result_dir = result_dir.name
        self._log.info("Temporary dir is: %s", self.__result_dir)

        workers = list()
        for i in range(0, self.__process_num):
            process = _mp.Process(target=self._worker_main,
                                  name="Teacher #%d" % i)
            process.daemon = True
            workers.append(process)

        for process in workers:
            process.start()

        for worker in workers:
            worker.join()
        del workers
        self._log.info("Teaching is complete.")

        self._log.info("Looking for best network.")
        best_error = None
        best_weights = None
        while not self.__result_queue.empty():
            res = self.__result_queue.get()
            if best_error is None or res["error"] < best_error:
                arrs = _np.load(res["filename"])
                weights = arrs["weights"]
                arrs.close()
                best_error = res["error"]
                best_weights = weights

        self._log.info("Cleaning up a temporary directory.")
        result_dir.cleanup()

        sec = time.time() - start_time
        self._log.info("Teaching took %.5f seconds(%.2f min).", sec, sec / 60)
        return best_weights, best_error

    def _worker_main(self):
        if self._logging_config is not None:
            logging_config = self._logging_config
            logging_config["format"] = "[{}] {}".format(_mp.current_process().name,
                                                        logging_config["format"])
            _log.basicConfig(**logging_config)
        else:
            _log.disable(_log.CRITICAL)
        # Log instance can not be pickled that is why we need to create a new log.
        log = _log.getLogger("teaches.MultiprocessTeacher")

        network = snn.Perceptron.load_from_json(self.__net_json)
        simple_teacher = snn.SimpleTeacher(self.__opt_manager, self._logging_config)
        network, error = simple_teacher.teach(network,
                                              self.__test_data,
                                              self.__nb_epoch,
                                              self.__eps,
                                              separate_weights=self.__separate_weights)

        filename = _path.join(self.__result_dir, "tmp_weights_{}.npz".
                              format(_mp.current_process().pid))
        _np.savez_compressed(filename, weights=network.get_weights())
        log.info("Saved current weights to \"%s\"", filename)
        result = dict()
        result["error"] = error
        result["filename"] = filename
        self.__result_queue.put(result)
        self.__result_queue.close()
        log.info("Done.")

    def __getstate__(self):
        # Must delete _log because it isn't pickable.
        state = dict(self.__dict__)
        del state["_log"]
        return state
