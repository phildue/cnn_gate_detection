import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Reshape, Lambda
from keras.optimizers import Adam

from modelzoo.backend.tensor.ConcatMeta import ConcatMeta
from modelzoo.backend.tensor.metrics import Loss
from modelzoo.models.Net import Net
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder


class GateNetSingle(Net):

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
                 img_shape=(52, 52),
                 grid=(1, 1),
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
        # only predict one set of boxes
        input = Input((w, h, 3))
        conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
        norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(alpha=0.1)(norm1)
        pool1 = MaxPooling2D((2, 2))(act1)
        # 26
        conv2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool1)
        norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(alpha=0.1)(norm2)
        pool2 = MaxPooling2D((2, 2))(act2)
        # 13
        conv3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool2)
        norm3 = BatchNormalization()(conv3)
        act3 = LeakyReLU(alpha=0.1)(norm3)
        pool3 = MaxPooling2D((2, 2))(act3)
        # 7
        conv3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool3)
        norm3 = BatchNormalization()(conv3)
        act3 = LeakyReLU(alpha=0.1)(norm3)
        pool3 = MaxPooling2D((2, 2))(act3)
        # 3
        conv4 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(pool3)
        norm4 = BatchNormalization()(conv4)
        act4 = LeakyReLU(alpha=0.1)(norm4)
        pool4 = MaxPooling2D((2, 2))(act4)
        # 1
        final = Conv2D(n_boxes * (n_polygon + 1), kernel_size=(1, 1), strides=(1, 1))(
            pool4)
        reshape = Reshape((-1, n_polygon + 1))(final)
        predictions = Lambda(self.net2y, (-1, n_polygon + 1))(reshape)

        meta_t = K.constant(GateNetEncoder.generate_anchors(self.norm, self.grid, self.anchors, self.n_polygon),
                            K.tf.float32)

        out = ConcatMeta((K.shape(predictions)), meta_t)(predictions)
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
        pred_xy = K.sigmoid(netout[:, :, 1:3])
        pred_wh = K.exp(netout[:, :, 3:self.n_polygon])
        pred_c = K.sigmoid(netout[:, :, :1])

        return K.concatenate([pred_xy, pred_wh, pred_c], -1)