from pathlib import Path

import tensorflow as tf
from keras import backend as K, Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape, Lambda

from modelzoo.backend.tensor.gatenet.GateNetV8 import GateNetV8
from modelzoo.models.gatenet.GateNet import GateNet
from utils.workdir import cd_work
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
# manually put back imported modules
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

cd_work()
in_file = 'logs/gatev8_mixed/model.h5'
out_path = 'logs/gatev8_mixed/'
out_name = out_path + 'model'
quantize = False
num_outputs = 1

"""
Prepare model
"""
K.set_learning_phase(0)
model = GateNet.v8(weight_file=in_file)

net_model = model.net.backend

netin = tf.placeholder(name="img", dtype=tf.float32, shape=model.input_shape)

num_output = num_outputs
netout = [None] * num_output
netout_names = [None] * num_output
for i in range(num_output):
    netout_names[i] = str(i)
    netout[i] = tf.identity(net_model.outputs[i], name=netout_names[i])
print('output nodes names are: ', netout_names)

sess = K.get_session()

"""
Convert variables to constants
"""
if quantize:
    from tensorflow.tools.graph_transforms import TransformGraph

    transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], netout_names, transforms)
    constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, netout_names)
else:
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), netout_names)


"""
Save pb graph
"""
graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name+'.pb', as_text=False)
print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name+'.pb')))

graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name+'.pbtext', as_text=True)
print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name+'.pbtext')))


"""
Convert to TFLite
"""

tflite_model = tf.contrib.lite.toco_convert(constant_graph, [netin], netout)
open(out_name+'.tflite', "wb").write(tflite_model)

print('Saved tflite model at: ', str(Path(out_name+'.tflite')))
