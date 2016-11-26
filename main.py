import logging
import os.path
import sys
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np

from SNN import SNN, load_from_file, Layer, OptSNN
from core import neuron_functions as nf
from teachers.MultiProcessTeacher import MultiProcessTeacher
from utils.datagen import DataGen
from utils.fast_nn import make_neurons, make_layer

if __name__ == "__main__":

    format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging_config = {"format": format, "level": logging.NOTSET}
    logging.basicConfig(**logging_config)
    log = logging.getLogger("main")

    input_count = 3
    output_count = 1

    weight_bounds = (-30.0, 30.0)
    func_bounds = (-30.0, 30.0)


    def f(x):
        k = 9.980493678923573
        return k + x[0] + x[1] + x[2]

    datagen = DataGen(f, input_count, [[-5.0, 5.0]], error=0)
    if len(sys.argv[1:]) == 0:
        snn = SNN(weight_bounds, func_bounds)
        l = Layer()
        l.add_neurons(make_neurons("simple_sigmoid", input_count, 2, 1))
        l.add_neurons(make_neurons("const", input_count, 2))
        snn.add_layer(l)
        snn.add_layer(make_layer("linear", snn.net_output_len(), 1))

        x, y = datagen.next(3000)
        snn.set_test_inputs(x, y)

        # Measuring time of snn.error function
        weights = snn.get_weights()
        t = timeit.Timer(lambda: snn.error(), "gc.enable()")
        times = 1
        seconds = t.timeit(times)
        seconds /= times
        log.info("Error of snn measured.")

        # Measuring time of opt snn.error function
        optsnn = OptSNN(snn)
        log.info("Opt snn created")
        optsnn.set_test_data(x, y)
        weights = optsnn.get_weights()
        t = timeit.Timer(lambda: optsnn.error())
        opt_seconds = t.timeit(times)
        opt_seconds /= times

        log.info("Execution time of error: %f sec(%5.0f ms)", seconds, seconds * 1000)
        log.info("Execution time of opt error: %f sec(%5.0f ms)", opt_seconds, opt_seconds * 1000)
        log.info("Start error: %f", snn.error())
        log.info("Start opt error: %f", optsnn.error())

        optimizers = [{"name": "coordinate_descent", "params": {"maxIter": 2000, "h": 3}},
                      {"name": "gradient_descent", "params": {"maxIter": 1000, "h": 2, "h_mul": 0.7}}]
        teacher = MultiProcessTeacher(optimizers,
                                      snn.to_json(False),
                                      {"x": x, "y": y},
                                      process_num=2,
                                      logging_config=None)
        error_history = open("error_history.csv", "w")
        best_result = None

        while True:
            result = teacher.teach()
            error_history.write(str(result["error"]))
            error_history.write("\n")
            error_history.flush()
            log.info("Current error: %f, best error: %s", result["error"],
                     None if best_result is None else best_result["error"])
            if best_result is None or result["error"] < best_result["error"]:
                best_result = result
                log.info("New best result: %f, snn: %f", best_result["error"],
                         snn.error(best_result["weights"]))
                snn.set_weights(best_result["weights"])
                snn.save_to_file("net.json")
            restart_time = 2
            log.info("Restarting in %d seconds.", restart_time)
            time.sleep(restart_time)

    elif os.path.exists(sys.argv[1]):
        snn = load_from_file(sys.argv[1])
        x, y = datagen.next(100)
        error_history = np.loadtxt("error_history.csv", dtype=np.float64, delimiter=",")

        plt.figure()
        plt.plot(error_history, "r-", label="Error")
        plt.legend(loc="upper right")
        plt.savefig("error_history.png")
        plt.cla()
        plt.plot(y, "b-", label="Good")
        plt.plot(snn.multi_input(x), "r-", label="Net")
        plt.legend(loc="upper right")
        snn.set_test_inputs(x, y)
        plt.title("MSE: {}".format(snn.error()))
        plt.savefig("data.png")

        #Draw neuron functions.
        if not os.path.exists("neurons_funcs"):
            os.mkdir("neurons_funcs")
        x = np.linspace(-1, 1, 100)
        l = 1
        for layer in snn.layers():
            n = 1
            for neuron in layer:
                plt.cla()
                if neuron.f_len() == 0:
                    y = nf.get(neuron.func_name())(x)
                else:
                    y = nf.get(neuron.func_name())(x, neuron.get_func_weights())
                plt.plot(x, y, "b-")
                if neuron.f_len() == 0:
                    plt.title("Layer: %d, neuron: %d, func name: %s" % (l, n, neuron.func_name()))
                else:
                    plt.title("Layer: %d, neuron: %d, func name: %s\nfunc_weight: %s" %
                              (l, n, neuron.func_name(), str(neuron.get_func_weights())))
                plt.savefig("neurons_funcs/layer_{}_neuron_{}.png".format(l, n))
                n += 1
            l += 1

        print("Done!")
