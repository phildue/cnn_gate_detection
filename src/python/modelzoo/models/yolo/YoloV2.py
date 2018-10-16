import keras.backend as K
from keras import Model, Input
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Activation, Lambda, Reshape
from keras.layers import concatenate
from keras.optimizers import Adam
from tensorflow import space_to_depth

from modelzoo.metrics.Loss import Loss
from modelzoo.models.Net import Net


class YoloV2(Net):
    @property
    def train_params(self):
        return self._params

    def predict(self, sample):
        return self._model.predict(sample)

    @staticmethod
    def space_to_depth2(input):
        return space_to_depth(input, 2)

    def compile(self, params=None, metrics=None):
        # default_sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9)

        if params is not None:
            self._params = params

        optimizer = Adam(self._params['lr'], self._params['beta_1'], self._params['beta_2'], self._params['epsilon'],
                         self._params['decay'])
        self._model.compile(
            loss=self.loss.compute,
            optimizer=optimizer,
            metrics=metrics)

    @property
    def backend(self):
        return self._model

    @backend.setter
    def backend(self, model):
        self._model = model

    def __init__(self,
                 loss: Loss,
                 norm=(416, 416),
                 grid=(13, 13),
                 n_boxes=5,
                 n_classes=20,
                 anchors=None,
                 weight_file=None):

        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        self.grid = grid
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.anchors = anchors
        self.norm = norm
        self.loss = loss

        input = Input(shape=(norm[0], norm[1], 3))
        # Layer 1
        with K.name_scope('layer_1'):
            net = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 2
        with K.name_scope('layer_2'):
            net = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 3
        with K.name_scope('layer_3'):
            net = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 4
        with K.name_scope('layer_4'):
            net = Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 5
        with K.name_scope('layer_5'):
            net = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 6
        with K.name_scope('layer_6'):
            net = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 7
        with K.name_scope('layer_7'):
            net = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer
        with K.name_scope('layer_8'):
            net = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 9
        with K.name_scope('layer_9'):
            net = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 10
        with K.name_scope('layer_10'):
            net = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 11
        with K.name_scope('layer_11'):
            net = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 12
        with K.name_scope('layer_12'):
            net = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 13
        with K.name_scope('layer_13'):
            net = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        fork = net

        net = MaxPooling2D(pool_size=(2, 2))(net)

        # Layer 14
        with K.name_scope('layer_14'):
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 15
        with K.name_scope('layer_15'):
            net = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 16
        with K.name_scope('layer_16'):
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 17
        with K.name_scope('layer_17'):
            net = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 18
        with K.name_scope('layer_18'):
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 19
        with K.name_scope('layer_19'):
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 20
        with K.name_scope('layer_20'):
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # route -9

        # Layer 21
        with K.name_scope('layer_21'):
            fork = Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False)(fork)
            fork = BatchNormalization()(fork)
            fork = LeakyReLU(alpha=0.1)(fork)
            # stride=2
            fork = Lambda(self.space_to_depth2)(fork)

        # Layer 22
        # [route]
        # layers=-1,-4
        with K.name_scope('layer_22'):
            join = concatenate([fork, net])
            net = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(join)
            net = BatchNormalization()(net)
            net = LeakyReLU(alpha=0.1)(net)

        # Layer 23
        with K.name_scope('layer_23'):
            net = Conv2D(n_boxes * (n_classes + 5), (1, 1), strides=(1, 1), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)

        net = Activation('linear')(net)
        net = Reshape((grid[0], grid[1], n_boxes, 4 + 1 + n_classes))(net)

        net = Lambda(self.net2y, (grid[0], grid[1], n_boxes, 5 + n_classes))(net)
        net = Reshape((grid[0] * grid[1] * n_boxes, 5 + n_classes))(net)

        self._model = Model(input, net)

        if weight_file is not None:
            self._model.load_weights(weight_file)

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
