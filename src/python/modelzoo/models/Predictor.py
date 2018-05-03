from abc import abstractmethod

from modelzoo.backend.tensor.metrics.Loss import Loss
from modelzoo.models.Decoder import Decoder
from modelzoo.models.Encoder import Encoder
from modelzoo.models.Net import Net
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image


class Predictor:
    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def compile(self, params=None, metrics=None):
        self.net.compile(params, metrics)

    def predict(self, sample):
        if isinstance(sample, list):
            predictions = self._predict_batch(sample)
            labels = [BoundingBox.to_label(b) for b in predictions]
            return labels
        else:
            prediction = self._predict_sample(sample)
            label = BoundingBox.to_label(prediction)
            return label

    def _predict_sample(self, sample: Image):

        sample_t = self.preprocessor.preprocess(sample)
        netout = self._model.predict(sample_t)

        predictions = self.postprocessor.postprocess(netout)[0]

        return predictions

    def _predict_batch(self, sample: [Image]):
        sample_t = self.preprocessor.preprocess_batch(sample)

        netout = self._model.predict(sample_t)

        predictions = self.postprocessor.postprocess(netout)

        return predictions

    def __init__(self, preprocessor, postprocessor, net: Net, loss: Loss, encoder: Encoder, decoder: Decoder):
        self._decoder = decoder
        self._encoder = encoder
        self._loss = loss
        self._model = net
        self._postprocessor = postprocessor
        self._preprocessor = preprocessor

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def postprocessor(self):
        return self._postprocessor

    @property
    def net(self):
        return self._model

    @property
    def loss(self):
        return self._loss
