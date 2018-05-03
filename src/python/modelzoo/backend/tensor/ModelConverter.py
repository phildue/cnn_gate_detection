import tensorflow as tf

from modelzoo.models.Predictor import Predictor


class ModelConverter:

    @staticmethod
    def convert(model: Predictor, path, filename=None):
        if not filename:
            filename = model.net.__class__.__name__
        img = tf.placeholder(name="img", dtype=tf.float32, shape=model.input_shape)
        out = tf.identity(model.output_shape, name="out")
        with tf.Session() as sess:
            tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
            open(path + filename + ".tflite", "wb").write(tflite_model)
