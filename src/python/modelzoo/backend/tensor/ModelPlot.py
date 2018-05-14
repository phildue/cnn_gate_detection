from keras.utils import plot_model

from modelzoo.models.Predictor import Predictor


class ModelPlot:

    def __init__(self, model: Predictor, show_shapes=False, show_layer_names=False):
        self.show_layer_names = show_layer_names
        self.show_shapes = show_shapes
        self.model = model

    def save(self, filename: str):
        plot_model(self.model.net.backend, filename, self.show_shapes, self.show_layer_names)
