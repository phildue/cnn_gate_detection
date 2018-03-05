from abc import abstractmethod

import keras.backend as K
from keras.optimizers import Adam, SGD

from modelzoo.backend.tensor.metrics.Loss import Loss
from modelzoo.models.Net import Net


class SSDNet(Net):
    def __init__(self,
                 loss: Loss,
                 weight_file=None,
                 ):

        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        self.loss = loss

        self._model, self._predictor_sizes = self.build_model()
        if weight_file is not None:
            self._model.load_weights(weight_file, by_name=True)

    @abstractmethod
    def build_model(self):
        return None, None
        pass

    @property
    def predictor_sizes(self):
        return self._predictor_sizes

    def compile(self, params=None, metrics=None):

        if params is not None:
            self._params = params

        if self._params['optimizer'] is 'SGD':
            optimizer = SGD(lr=self._params['lr'], momentum=self._params['momentum'], decay=self._params['decay'])
        else:
            optimizer = Adam(self._params['lr'], self._params['beta_1'], self._params['beta_2'],
                             self._params['epsilon'],
                             self._params['decay'])

        self.backend.compile(optimizer=optimizer, loss=self.loss.compute,
                             metrics=metrics)

    def predict(self, sample):
        return self.backend.predict(sample)

    @property
    def backend(self):
        return self._model

    def compute_loss(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="y_pred")
        loss_t = self.loss.compute(y_true=y_true_k, y_pred=y_pred_k)
        return loss_t.eval()

    @property
    def train_params(self):
        return self._params
