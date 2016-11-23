from SNN import SNN, load_from_file
from utils.fast_nn import make_neurons, make_layer
from utils.datagen import DataGen
from teachers.MultiProcessTeacher import MultiProcessTeacher
import neuron_functions as nf

import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import timeit
import logging


if __name__ == "__main__":

    format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging_config = {"format": format, "level": logging.NOTSET}
    logging.basicConfig(**logging_config)
    log = logging.getLogger("main")

    input_count = 3
    output_count = 1

    weight_bounds = (-10, 10)
    func_bounds = (-10, 10)


    def f(x):
        k = 9.980493678923573
        return k * x[0] * x[1] * x[2]

    datagen = DataGen(f, input_count, [[-5.0, 5.0]], error=0)
    if len(sys.argv[1:]) == 0:
        snn = SNN(weight_bounds, func_bounds)
        snn.add_layer(make_layer("simple_sigmoid", input_count, 5, 1))
        snn.add_layer(make_layer("simple_sigmoid", snn.net_output_len(), 5, 1))
        snn.add_layer(make_layer("linear", snn.net_output_len(), 1))

        x, y = datagen.next(300)

        # Measuring time of snn.error function
        snn.set_test_inputs(x, y)
        weights = snn.get_weights()
        t = timeit.Timer(lambda: snn.error(weights), "gc.enable()")
        seconds = t.timeit(1)
        log.info("Execution time of error: %f sec(%5.0f ms)",seconds, seconds * 1000)
        del t

        optimizers = [{"name": "coordinate_descent", "params": {"maxIter": 100, "h": 3}},
                      {"name": "gradient_descent", "params": {"maxIter": 100, "h": 0.5, "h_mul": 0.5}}]
        teacher = MultiProcessTeacher(optimizers,
                                      snn.to_json(),
                                      {"x": x, "y": y},
                                      process_num=7,
                                      logging_config=logging_config)
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
                log.info("New best result: %f", best_result["error"])
                snn.set_weights(best_result["weights"])
                snn.save_to_file("net.json")

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
