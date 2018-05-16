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
model = GateNet.v8(weight_file=in_file)

K.set_learning_phase(0)

input = Input((416, 416, 3))
conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
norm1 = BatchNormalization()(conv1)
act1 = LeakyReLU(alpha=0.1)(norm1)
pool1 = MaxPooling2D((2, 2))(act1)
# 208
conv2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool1)
norm2 = BatchNormalization()(conv2)
act2 = LeakyReLU(alpha=0.1)(norm2)
pool2 = MaxPooling2D((2, 2))(act2)
# 104
conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool2)
norm3 = BatchNormalization()(conv3)
act3 = LeakyReLU(alpha=0.1)(norm3)
pool3 = MaxPooling2D((2, 2))(act3)
# 52
conv4 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool3)
norm4 = BatchNormalization()(conv4)
act4 = LeakyReLU(alpha=0.1)(norm4)
pool4 = MaxPooling2D((2, 2))(act4)
# 26
conv5 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool4)
norm5 = BatchNormalization()(conv5)
act5 = LeakyReLU(alpha=0.1)(norm5)
pool5 = MaxPooling2D((2, 2))(act5)

conv6 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool5)
norm6 = BatchNormalization()(conv6)
act6 = LeakyReLU(alpha=0.1)(norm6)

conv7 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act6)
norm7 = BatchNormalization()(conv7)
act7 = LeakyReLU(alpha=0.1)(norm7)

conv8 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act7)
norm8 = BatchNormalization()(conv8)
act8 = LeakyReLU(alpha=0.1)(norm8)

conv9 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act8)
norm9 = BatchNormalization()(conv9)
act8 = LeakyReLU(alpha=0.1)(norm9)

final = Conv2D(5 * (4 + 1), kernel_size=(1, 1), strides=(1, 1))(
    act8)
reshape = Reshape((13 * 13 * 5, 4 + 1))(final)
out = Lambda(model.net.net2y, (13 * 13 * 5, 4 + 1))(reshape)

net_model = Model(input, out)
net_model.load_weights(in_file)
# net_model = Model(input,pool2)

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
