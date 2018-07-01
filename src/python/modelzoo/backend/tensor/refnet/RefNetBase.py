from keras import Input, Model
from keras.layers import Conv2D, Reshape, Lambda
from keras.optimizers import Adam
from spp.RoiPoolingConv import RoiPoolingConv
import keras.backend as K
from modelzoo.backend.tensor.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.backend.tensor.gatenet.Netout import Netout
from modelzoo.backend.tensor.layers import create_layer
from modelzoo.backend.tensor.refnet.ConcatMeta import ConcatMeta
from modelzoo.models.Net import Net
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
import numpy as np


class RefNetBase(Net):

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

    def predict(self, sample, roi=None):
        if roi is None:
            roi = np.array([[self.norm[0] / 2, self.norm[1] / 2, self.norm[0], self.norm[1]]])
        self._model.predict([sample, roi])

    @property
    def backend(self):
        return self._model

    @property
    def train_params(self):
        return self._params

    def __init__(self, norm, n_rois, anchors, n_polygon, architecture, n_boxes, weight_file=None,
                 loss=GateDetectionLoss(), pool_size=4):
        self.loss = loss
        self.norm = norm

        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        h, w = norm
        inimg = Input((h, w, 3))
        inroi = Input((n_rois, 4))

        net = RoiPoolingConv(pool_size=pool_size, num_rois=n_rois)([inimg, inroi])

        def patch_flatten(x):
            return K.reshape(x, (-1, pool_size, pool_size, 3))

        net = Lambda(patch_flatten, output_shape=(pool_size, pool_size, 3))(net)
        grid = pool_size, pool_size
        for config in architecture:
            net = create_layer(net, config)
            if 'pool' in config['name']:
                size = config['size']
                grid = int(grid[0] / size[0]), int(grid[1] / size[1])
        self.grid = [grid]
        final = Conv2D(n_boxes * (n_polygon + 1), kernel_size=(1, 1), strides=(1, 1))(
            net)
        reshape = Reshape((-1, n_polygon + 1))(final)
        predictions = Netout(K.shape(reshape))(reshape)

        def patch_expand(x):
            return K.reshape(x, (-1, n_rois, grid[0] * grid[1] * n_boxes, n_polygon + 1))

        predictions_exp = Lambda(patch_expand, output_shape=(n_rois, grid[0] * grid[1] * n_boxes, n_polygon + 1))(
            predictions)

        meta_t = K.constant(GateNetEncoder.generate_anchors(norm, self.grid, anchors, n_polygon),
                            K.tf.float32)

        augmented = ConcatMeta((K.shape(predictions_exp)), meta_t, inroi)(predictions_exp)

        # netout = Reshape((n_rois*grid[0] * grid[1] * n_boxes, n_polygon + 1 + 4 + 4))(augmented)

        model = Model([inimg, inroi], augmented)

        if weight_file is not None:
            model.load_weights(weight_file)
        self._model = model
