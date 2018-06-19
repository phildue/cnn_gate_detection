from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dense, Flatten, Reshape
from keras.optimizers import Adam

from modelzoo.backend.tensor.CropLoss import CropLoss
from modelzoo.models.Net import Net


class CropNet2L(Net):

    def __init__(self, loss: CropLoss, input_shape=(52, 52)):
        self.loss = loss
        self.grid_shape = int(input_shape[0] / 2 ** 2), int(input_shape[1] / 2 ** 2)
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}
        h, w = input_shape
        netin = Input((h, w, 3))

        conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(netin)
        norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(alpha=0.1)(norm1)

        pool1 = MaxPooling2D()(act1)

        conv2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool1)
        norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(alpha=0.1)(norm2)

        pool2 = MaxPooling2D()(act2)

        conv3 = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool2)
        norm3 = BatchNormalization()(conv3)
        reshape = Flatten()(norm3)
        dense = Dense(self.grid_shape[0] * self.grid_shape[1])(reshape)
        out = Reshape(self.grid_shape)(dense)

        self._model = Model(netin, out)

    def compile(self, params=None, metrics=None):
        # default_sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9)
        if params is not None:
            self._params = params

        optimizer = Adam(self._params['lr'], self._params['beta_1'], self._params['beta_2'], self._params['epsilon'],
                         self._params['decay'])

        self._model.compile(
            loss=self.loss.compute,
            optimizer=optimizer,
            metrics=metrics
        )

    def predict(self, sample):
        pass

    @property
    def backend(self):
        return self._model

    @property
    def train_params(self):
        return self._params
