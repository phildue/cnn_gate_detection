from abc import abstractmethod

import keras.backend as K
import numpy as np
from keras.optimizers import Adam, SGD

from modelzoo.metrics.Loss import Loss
from modelzoo.models.Net import Net


class SSDNet(Net):
    def __init__(self,
                 img_shape,
                 variances,
                 scales,
                 aspect_ratios,
                 loss: Loss,
                 ):

        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.variances = variances
        self.img_shape = img_shape
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        self.loss = loss

    @property
    @abstractmethod
    def anchors(self):
        pass

    @property
    @abstractmethod
    def backend(self):
        pass

    @backend.setter
    @abstractmethod
    def backend(self, model):
        pass

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

    def compute_loss(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="y_pred")
        loss_t = self.loss.compute(y_true=y_true_k, y_pred=y_pred_k)
        return loss_t.eval()

    @property
    def train_params(self):
        return self._params

    def _generate_meta_t(self, anchors):
        """
        Generate tensor with meta information for bounding box. That way
        the label can be decoded without external information.
        - var_t: variance for decode/encode
        - anchor_cxy: anchor center
        - anchor_wh: anchor width/height
        - dummy: the encoded label contains idx of true box it has been assigned to here we just output nan
        :return: tensor(#boxes,4+2+2+1): [var_t, anchor_cxy anchor_wh, dummy]
        """
        var_t = K.constant([self.variances])
        var_t = K.tile(var_t, (anchors.shape[0], 1))

        anchor_wh = anchors[:, 2:] / K.np.array([self.img_shape[1], self.img_shape[0]])
        anchor_cxy = anchors[:, :2] / K.np.array([self.img_shape[1], self.img_shape[0]])

        meta_t = K.concatenate([var_t, anchor_cxy, anchor_wh, K.np.nan * K.ones((anchors.shape[0], 1))], -1)

        return meta_t

    def generate_anchor_t(self,
                          feature_map_size,
                          aspect_ratios,
                          scale, next_scale):

        """
        Compute an array of the spatial positions and sizes of the anchor boxes for one particular classification
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        :param feature_map_size:  tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
        :param aspect_ratios: A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
        :param scale: A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
        :return: tensor(feature_map_h*feature_map_w*#boxes,4)
        """

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_shape[:1])
        n_boxes = len(aspect_ratios)
        if 1.0 in aspect_ratios:
            n_boxes += 1

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_shape[0] / feature_map_size[0]
        cell_width = self.img_shape[1] / feature_map_size[1]
        cx = np.linspace(cell_width / 2, self.img_shape[1] - cell_width / 2, feature_map_size[1])
        cy = np.linspace(cell_height / 2, self.img_shape[0] - cell_height / 2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy

        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for i, ar in enumerate(aspect_ratios):
            if ar == 1.0:
                scale_sqrt = np.sqrt(scale * next_scale)
                w = scale_sqrt * size * np.sqrt(ar)
                h = scale_sqrt * size / np.sqrt(ar)
                wh_list.append((w, h))

            w = scale * size * np.sqrt(ar)
            h = scale * size / np.sqrt(ar)
            wh_list.append((w, h))

        wh_list = np.array(wh_list)

        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        boxes_tensor = np.reshape(boxes_tensor, (-1, 4))

        return boxes_tensor

    def generate_anchors_t(self, predictor_sizes):
        """
        Generates a tensor that contains the anchor box coordinates.
        The shape and content is determined by the number of predictor layers used in the
        model architecture and the amount of anchor boxes for each predictor.

        Returns:
            A Numpy array of shape `(#boxes, 4)` [cx,cy,w,h]
        """

        boxes_tensor = []
        for i in range(len(predictor_sizes)):
            boxes = self.generate_anchor_t(feature_map_size=predictor_sizes[i],
                                           aspect_ratios=self.aspect_ratios[i],
                                           scale=self.scales[i],
                                           next_scale=self.scales[i + 1])
            boxes_tensor.append(boxes)

        return np.concatenate(boxes_tensor, axis=0)
