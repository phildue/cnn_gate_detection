import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Reshape, Lambda, AtrousConv2D
from keras.optimizers import Adam

from modelzoo.backend.tensor.metrics import Loss
from modelzoo.models.Net import Net


class GateNetV28(Net):

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
                 img_shape=(416, 416),
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
        # Atrous conv
        input = Input((w, h, 3))
        conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
        norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(alpha=0.1)(norm1)
        pool1 = MaxPooling2D((2, 2))(act1)
        # 208
        conv2 = AtrousConv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                             atrous_rate=(9, 9))(pool1)
        norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(alpha=0.1)(norm2)
        pool2 = MaxPooling2D((2, 2))(act2)
        # 104
        conv3 = AtrousConv2D(64, kernel_size=(3, 3), atrous_rate=(9, 9), strides=(1, 1), padding='same',
                             use_bias=False)(pool2)
        norm3 = BatchNormalization()(conv3)
        act3 = LeakyReLU(alpha=0.1)(norm3)
        pool3 = MaxPooling2D((2, 2))(act3)
        # 52
        conv4 = AtrousConv2D(64, kernel_size=(3, 3), atrous_rate=(9, 9), strides=(1, 1), padding='same',
                             use_bias=False)(pool3)
        norm4 = BatchNormalization()(conv4)
        act4 = LeakyReLU(alpha=0.1)(norm4)
        pool4 = MaxPooling2D((2, 2))(act4)
        # 26
        conv5 = AtrousConv2D(64, kernel_size=(6, 6), atrous_rate=(9, 9), strides=(1, 1), padding='same',
                             use_bias=False)(pool4)
        norm5 = BatchNormalization()(conv5)
        act5 = LeakyReLU(alpha=0.1)(norm5)
        pool5 = MaxPooling2D((2, 2))(act5)
        # 13
        conv6 = Conv2D(64, kernel_size=(6, 6), strides=(1, 1), padding='same', use_bias=False)(pool5)
        norm6 = BatchNormalization()(conv6)
        act6 = LeakyReLU(alpha=0.1)(norm6)

        final = Conv2D(n_boxes * (n_polygon + 1), kernel_size=(1, 1), strides=(1, 1))(
            act6)
        reshape = Reshape((self.grid[0] * self.grid[1] * self.n_boxes, n_polygon + 1))(final)
        out = Lambda(self.net2y, (grid[0] * grid[1] * n_boxes, n_polygon + 1))(reshape)

        model = Model(input, out)

        if weight_file is not None:
            model.load_weights(weight_file)

        self._model = model

    def net2y(self, netout):
        """
        Adapt raw network output. (Softmax, exp, sigmoid, anchors)
        :param netout: Raw network output
        :return: y as fed for learning
        """
        netout = K.reshape(netout, (-1, self.grid[0] * self.grid[1], self.n_boxes, self.n_polygon + 1))
        pred_xy = K.sigmoid(netout[:, :, :, :2])
        pred_wh = K.exp(netout[:, :, :, 2:self.n_polygon]) * K.reshape(K.constant(self.anchors),
                                                                       [1, 1, self.n_boxes, self.n_polygon - 2])
        pred_c = K.sigmoid(netout[:, :, :, -1])

        return K.concatenate([pred_xy, pred_wh, K.expand_dims(pred_c, -1)], 3)
