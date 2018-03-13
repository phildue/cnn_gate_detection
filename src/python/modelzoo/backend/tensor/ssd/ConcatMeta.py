from keras.engine.topology import Layer
import keras.backend as K


class ConcatMeta(Layer):
    def __init__(self, output_dim, meta_t, **kwargs):
        self.meta_t = meta_t
        self.output_dim = output_dim
        super(ConcatMeta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ConcatMeta, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        meta_t = K.expand_dims(self.meta_t, 0)
        meta_t = K.tile(meta_t, (self.output_dim[0], 1, 1))
        return K.concatenate([x, meta_t])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] + 9
