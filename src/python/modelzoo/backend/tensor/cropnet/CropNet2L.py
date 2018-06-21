from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dense, Flatten, Reshape, Activation
from keras.optimizers import Adam

from modelzoo.backend.tensor.CropGridLoss import CropGridLoss
from modelzoo.backend.tensor.layers import create_layer
from modelzoo.models.Net import Net


class CropNet2L(Net):

    def __init__(self, architecture, loss=CropGridLoss(), input_shape=(52, 52)):
        self.loss = loss
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}
        h, w = input_shape
        if architecture is None:
            architecture = [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1}]
        netin = Input((h, w, 3))
        net = netin
        grid = h,w
        for config in architecture:
            net = create_layer(net, config)
            if 'pool' in config['name']:
                size = config['size']
                grid = int(grid[0] / size[0]), int(grid[1] / size[1])

        self.grid = grid
        flat = Flatten()(net)
        dense = Dense(self.grid[0] * self.grid[1])(flat)
        act = Activation('softmax')(dense)
        netout = Reshape(self.grid)(act)

        self._model = Model(netin, netout)

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
