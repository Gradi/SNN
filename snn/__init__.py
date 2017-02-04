import snn.core.snn_core as _snn
import snn.utils.fast_nn as _fnn

import snn.teachers.MultiProcessTeacher as _MPT
import snn.teachers.SimpleTeacher as _ST

import snn.optimizers.optimizer_manager as _opt_mgr

import snn.utils.datagen as _gen
import snn.network.perceptron_numpy as _perceptron

Neuron = _snn.Neuron
Layer  = _snn.Layer
Perceptron = _perceptron.PerceptronNumpy

make_neurons = _fnn.make_neurons
make_layer = _fnn.make_layer

MultiProcessTeacher = _MPT.MultiProcessTeacher
SimpleTeacher = _ST.SimpleTeacher
OptManager = _opt_mgr.OptManager

DataGen = _gen.DataGen
