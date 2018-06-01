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
if __name__ == '__main__':

    input_layer = tf.placeholder(tf.float32, [1, 480, 640, 3],name='Image')
    quantize = False
    out_name = 'build/test.tflite'
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
    )
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=[2, 2],
        strides=2,
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
    )
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=[2, 2],
        strides=2,
    )

    predictions = tf.layers.conv2d(
        pool2,
        filters=5,
        kernel_size=[16, 16],
        strides=16,
        activation=tf.nn.relu,
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