from keras.engine.topology import Layer
import keras.backend as K


class Netout(Layer):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
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
        pred_c = x[:, :, 0:1]
        pred_class = x[:, :, 1:self.n_classes]
        pred_xy = K.sigmoid(x[:, :, -4:-2])
        pred_wh = K.exp(x[:, :, -2:])

        return K.concatenate([pred_c, pred_class, pred_xy, pred_wh], -1)

    def compute_output_shape(self, input_shape):
        return input_shape
