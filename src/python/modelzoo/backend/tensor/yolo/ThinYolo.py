import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Activation, Reshape, Lambda
from keras.optimizers import Adam

from modelzoo.backend.tensor.metrics.Loss import Loss
from modelzoo.models.Net import Net


class ThinYolo(Net):
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
                 norm=(416, 416),
                 grid=(13, 13),
                 n_boxes=5,
                 n_classes=20,
                 weight_file=None):

        self.loss = loss
        self.grid = grid
        self.norm = norm
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.anchors = anchors
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        input = Input(shape=(norm[0], norm[1], 3))

        with K.name_scope('layer_1'):
            conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
            norm1 = BatchNormalization()(conv1)
            act1 = LeakyReLU(alpha=0.1)(norm1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

        # Layer 2 - 5
        net = pool1
        for i in range(0, 4):
            with K.name_scope('layer_' + str(i + 2)):
                net = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
                net = BatchNormalization()(net)
                net = LeakyReLU(alpha=0.1)(net)
                net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 6
        with K.name_scope('layer_6'):
            conv6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            norm6 = BatchNormalization()(conv6)
            act6 = LeakyReLU(alpha=0.1)(norm6)

        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(act6)

        # Layer 7 - 8
        net = pool3
        for i in range(0, 2):
            with K.name_scope('layer_' + str(i + 7)):
                net = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
                net = BatchNormalization()(net)
                net = LeakyReLU(alpha=0.1)(net)

        # Layer 9
        with K.name_scope('layer_' + str(9)):
            conv9 = Conv2D(n_boxes * (4 + 1 + n_classes), (1, 1), strides=(1, 1), kernel_initializer='he_normal')(
                net)
            act9 = Activation('linear')(conv9)
            reshape9 = Reshape((grid[0], grid[1], n_boxes, 4 + 1 + n_classes))(act9)

        lambda_reshape = Lambda(self.net2y, (grid[0], grid[1], n_boxes, 5 + n_classes))(reshape9)
        net = Reshape((grid[0] * grid[1] * n_boxes, 5 + n_classes))(lambda_reshape)

        model = Model(input, net)

        if weight_file is not None:
            model.load_weights(weight_file)

        self._model = model

    def net2y(self, netout):
        """
        Adapt raw network output. (Softmax, exp, sigmoid, anchors)
        :param netout: Raw network output
        :return: y as fed for learning
        """
        pred_xy = K.sigmoid(netout[:, :, :, :, :2])
        pred_wh = K.exp(netout[:, :, :, :, 2:4]) * K.reshape(K.constant(self.anchors), [1, 1, 1, self.n_boxes, 2])
        pred_c = K.sigmoid(netout[:, :, :, :, 4])
        pred_c = K.reshape(pred_c, [-1, self.grid[0], self.grid[1], self.n_boxes, 1])
        pred_class_likelihoods = K.softmax(netout[:, :, :, :, 5:]) * pred_c

        return K.concatenate([pred_xy, pred_wh, pred_c, pred_class_likelihoods], 4)
