# manually put back imported modules
import keras.backend as K
from keras import Model

from modelzoo.backend.tensor.graphconv import convert_model
from modelzoo.backend.tensor.gatenet.PostprocessLayer import PostprocessLayer
from modelzoo.models.ModelBuilder import ModelBuilder


class ModelConverter:

    def __init__(self, model_name, directory):
        self.model_name = model_name
        self.directory = directory

    def finalize(self, quantize=False):
        K.set_learning_phase(0)
        model = ModelBuilder.get_model(self.model_name, batch_size=1, src_dir=self.directory)

        backend = model.net.backend
        out = backend.layers[-1].output
        input = backend.layers[0].input
        postprocessed = PostprocessLayer()(out)
        inference_model = Model(input, postprocessed)
        sess = K.get_session()
        convert_model(sess, inference_model, out_name=self.directory + self.model_name, out_format=['tflite'],
                      quantize=quantize,input_shape=model.input_shape)
