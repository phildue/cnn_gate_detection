from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU

from modelzoo.models.Net import Net


class PDNet(Net):

    def __init__(self, img_shape: (int, int), n_polygon=4, n_boxes=845):
        w, h = img_shape
        input = Input(w, h)
        conv1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
        norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(alpha=0.1)(norm1)
        pool1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', use_bias=False)(act1)

        conv2 = Conv2D(128, kernel_size=(3, 3))(pool1)
        norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(alpha=0.1)(norm2)
        pool2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', use_bias=False)(act2)
        # TODO kernel size should be whole feature map of pool2
        out = Conv2D(1 + n_polygon + n_boxes, kernel_size=(3, 3), strides=(1, 1))(pool2)

        self._model = Model(input, out)

    def compile(self, params=None, metrics=None):
        pass

    def predict(self, sample):
        pass

    @property
    def backend(self):
        return self._model

    @property
    def train_params(self):
        pass
