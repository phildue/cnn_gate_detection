import keras.backend as K
from keras.engine.topology import Layer


class Netout(Layer):
    def __init__(self, polygon, **kwargs):
        self.polygon = polygon
        super(Netout, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Netout, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        """
        Adapt raw network output. (Softmax, exp, sigmoid, anchors)
        :param netout: Raw network output
        :return: y as fed for learning
        """
        pred_xy = K.sigmoid(x[:, :, :2])
        pred_wh = K.exp(x[:, :, 2:self.polygon])
        pred_c = x[:, :, -1:]

        return K.concatenate([pred_c, pred_xy, pred_wh], -1)

    def compute_output_shape(self, input_shape):
        return input_shape
