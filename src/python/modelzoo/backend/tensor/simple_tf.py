from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import graph_io
# manually put back imported modules
import tempfile
import subprocess

from utils.workdir import cd_work

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess
cd_work()


def conv_pool_norm(input):
    conv = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=[6, 6],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
    )
    pool = tf.layers.max_pooling2d(
        conv,
        pool_size=[2, 2],
        strides=2,
    )

    norm = tf.layers.batch_normalization(
        inputs=pool
    )

    return norm


def conv_norm(input, kernel_size):
    conv = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
    )

    norm = tf.layers.batch_normalization(
        inputs=conv
    )

    return norm


if __name__ == '__main__':

    input_layer = tf.placeholder(tf.float32, [1, 416, 416, 3], name='Image')
    quantize = False
    out_name = 'build/test'

    layer1 = conv_pool_norm(input_layer)
    layer2 = conv_pool_norm(layer1)
    layer3 = conv_pool_norm(layer2)
    layer4 = conv_pool_norm(layer3)
    layer5 = conv_pool_norm(layer4)
    layer6 = conv_norm(layer5, [6, 6])
    layer7 = conv_norm(layer6, [3, 3])
    layer8 = conv_norm(layer7, [3, 3])

    predictions = tf.layers.conv2d(
        layer8,
        filters=5 * 5,
        kernel_size=[1, 1],
        strides=1,
        activation=None,  # linear
        use_bias=False,
    )

    netout = [tf.identity(predictions, name='Predictions')]
    netout_names = ['Predictions']
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        """
        Convert variables to constants
        """
        if quantize:
            from tensorflow.tools.graph_transforms import TransformGraph

            transforms = ["quantize_weights", "quantize_nodes"]
            transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], netout_names, transforms)
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, transformed_graph_def, netout_names)
        else:
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), netout_names)

        """
        Save pb graph
        """
        # if 'pb' in out_format:
        #     graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name + '.pb', as_text=False)
        #     print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name + '.pb')))
        # if 'pbtext' in out_format:
        #     graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name + '.pbtext', as_text=True)
        #     print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name + '.pbtext')))
        #
        """
        Convert to TFLite
        """
        # if 'tflite' in out_format:
        tflite_model = tf.contrib.lite.toco_convert(constant_graph, [input_layer], netout)
        open(out_name + '.tflite', "wb").write(tflite_model)
        print('Saved tflite model at: ', str(Path(out_name + '.tflite')))
