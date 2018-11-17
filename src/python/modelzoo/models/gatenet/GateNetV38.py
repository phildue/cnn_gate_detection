import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Reshape
from keras.optimizers import Adam

from modelzoo.layers.ConcatMeta import ConcatMeta
from modelzoo.metrics import Loss
from modelzoo.models.Net import Net
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from modelzoo.models.gatenet.Netout import Netout


class GateNetV38(Net):

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

    @property
    def backend(self):
        return self._model

    @backend.setter
    def backend(self, model):
        self._model = model

    @property
    def train_params(self):
        return self._params

    def predict(self, sample):
        return self._model.predict(sample)

    def __init__(self, loss: Loss,
                 anchors,
                 img_shape=(104, 104),
                 grid=(13, 13),
                 n_boxes=5,
                 weight_file=None, n_polygon=4):

        self.loss = loss
        self.grid = grid
        self.norm = img_shape
        self.n_boxes = n_boxes
        self.anchors = anchors
        self.n_polygon = n_polygon
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        w, h = img_shape
        # like gatev10 but on 128x128
        input = Input((w, h, 3))
        conv1 = Conv2D(16, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(input)
        norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(alpha=0.1)(norm1)
        pool1 = MaxPooling2D((2, 2))(act1)
        # 56
        conv2 = Conv2D(32, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(pool1)
        norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(alpha=0.1)(norm2)
        pool2 = MaxPooling2D((2, 2))(act2)
        # 26
        conv3 = Conv2D(64, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(pool2)
        norm3 = BatchNormalization()(conv3)
        act3 = LeakyReLU(alpha=0.1)(norm3)
        pool3 = MaxPooling2D((2, 2))(act3)
        # 13
        conv4 = Conv2D(64, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(pool3)
        norm4 = BatchNormalization()(conv4)
        act4 = LeakyReLU(alpha=0.1)(norm4)

        conv5 = Conv2D(64, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(act4)
        norm5 = BatchNormalization()(conv5)
        act5 = LeakyReLU(alpha=0.1)(norm5)

        conv6 = Conv2D(64, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(act5)
        norm6 = BatchNormalization()(conv6)
        act6 = LeakyReLU(alpha=0.1)(norm6)

        conv7 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act6)
        norm7 = BatchNormalization()(conv7)
        act7 = LeakyReLU(alpha=0.1)(norm7)

        conv8 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act7)
        norm8 = BatchNormalization()(conv8)
        act8 = LeakyReLU(alpha=0.1)(norm8)

        final = Conv2D(n_boxes * (n_polygon + 1), kernel_size=(1, 1), strides=(1, 1))(
            act8)
        reshape = Reshape((self.grid[0][0] * self.grid[0][1] * self.n_boxes, n_polygon + 1))(final)
        predictions = Netout(K.shape(reshape))(reshape)

        meta_t = K.constant(GateNetEncoder.generate_encoding(self.norm, self.grid, self.anchors, self.n_polygon),
                            K.tf.float32)

        out = ConcatMeta((K.shape(predictions)), meta_t)(predictions)
        model = Model(input, out)

        if weight_file is not None:
            model.load_weights(weight_file)

        self._model = model

