from abc import abstractmethod

import keras.backend as K

from modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision
from modelzoo.backend.tensor.metrics.Metric import Metric
import numpy as np

class MetricYolo(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, n_boxes, grid, iou_thresh, n_classes, norm, iou_thresh_nms,
                 batch_size, confidence_levels=K.np.linspace(0, 1.0, 11)):
        self.confidence_levels = confidence_levels
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.norm = norm
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.grid = grid
        total_boxes = np.sum([grid[i][0] * grid[i][1] * n_boxes[i] for i in range(len(n_boxes))])
        self.map_adapter = AveragePrecision(iou_thresh,total_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t, anchors_t):
        coord_t_cx = coord_t[:, :, 0] * anchors_t[:, :, 2]
        coord_t_w = coord_t[:, :, 2] * anchors_t[:, :, 2]
        coord_t_cy = coord_t[:, :, 1] * anchors_t[:, :, 3]
        coord_t_h = coord_t[:, :, 3] * anchors_t[:, :, 3]

        coord_t_cx = coord_t_cx + anchors_t[:, :, 0]
        coord_t_cy = coord_t_cy + anchors_t[:, :, 1]

        coord_t_cy = self.norm[0] - coord_t_cy

        coord_t_xmin = coord_t_cx - coord_t_w / 2
        coord_t_ymin = coord_t_cy - coord_t_h / 2
        coord_t_xmax = coord_t_cx + coord_t_w / 2
        coord_t_ymax = coord_t_cy + coord_t_h / 2

        coord_t_xmin = K.expand_dims(coord_t_xmin, -1)
        coord_t_ymin = K.expand_dims(coord_t_ymin, -1)
        coord_t_xmax = K.expand_dims(coord_t_xmax, -1)
        coord_t_ymax = K.expand_dims(coord_t_ymax, -1)
        coord_dec_t = K.concatenate([coord_t_xmin, coord_t_ymin, coord_t_xmax, coord_t_ymax], -1)

        return coord_dec_t

    def _postprocess_truth(self, y_true):
        conf_t, class_t, coord_t, anchors_t = MetricYolo.split_t(y_true)

        coord_dec_t = self._decode_coord(coord_t, anchors_t)

        return conf_t, class_t, coord_dec_t

    @staticmethod
    def split_t(label_t):
        coord_t = label_t[:, :, -8:-4]
        conf_t = label_t[:, :, :1]
        class_t = label_t[:, :, 1:-8]
        anchors_t = label_t[:, :, -4:]

        return conf_t, class_t, coord_t, anchors_t

    def _postprocess_pred(self, y_pred):
        conf_t, class_t, coord_t, anchors_t = MetricYolo.split_t(y_pred)
        conf_pp_t = K.sigmoid(conf_t)
        class_pp_t = K.softmax(class_t)

        coord_dec_t = self._decode_coord(coord_t, anchors_t)

        class_nms_batch = self.map_adapter.non_max_suppression_batch(coord_dec_t,
                                                                     class_pp_t,
                                                                     self.batch_size,
                                                                     K.shape(anchors_t)[0],
                                                                     self.iou_thresh)

        return conf_pp_t, class_nms_batch, coord_dec_t
