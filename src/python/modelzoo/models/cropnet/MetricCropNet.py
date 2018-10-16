from abc import abstractmethod

import keras.backend as K

from modelzoo.metrics.AveragePrecision import AveragePrecision
from modelzoo.metrics.Metric import Metric


class MetricCropNet(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, n_boxes=5, grid=(13, 13), iou_thresh=0.4, norm=(416, 416), iou_thresh_nms=0.4,
                 batch_size=8, confidence_levels=K.np.linspace(0, 1.0, 11)):
        self.confidence_levels = confidence_levels
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.norm = norm
        self.n_boxes = n_boxes
        self.grid = grid
        self.map_adapter = AveragePrecision(iou_thresh, grid[0][1] * grid[0][0] * n_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t, anchors_t):
        coord_t_cx = coord_t[:, :, 0] * anchors_t[:, :, 2]
        coord_t_w = coord_t[:, :, 2] * anchors_t[:, :, 2]
        coord_t_cy = coord_t[:, :, 1] * anchors_t[:, :, 2]
        coord_t_h = coord_t[:, :, 2] * anchors_t[:, :, 2]

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
        coord_true_t = y_true[:, :, 1:4]
        conf_true_t = y_true[:, :, :1]
        anchors_t = y_true[:, :, 4:]

        coord_true_dec_t = self._decode_coord(coord_true_t, anchors_t)

        return coord_true_dec_t, conf_true_t

    def _postprocess_pred(self, y_pred):
        coord_pred_t = y_pred[:, :, 1:4]
        conf_pred_t = y_pred[:, :, :1]
        anchors_t = y_pred[:, :, 4:]

        coord_pred_dec_t = self._decode_coord(coord_pred_t, anchors_t)

        class_pred_nms_batch = self.map_adapter.non_max_suppression_batch(coord_pred_dec_t,
                                                                          conf_pred_t,
                                                                          self.batch_size,
                                                                          self.n_boxes * self.grid[0][0] * self.grid[0][
                                                                              1],
                                                                          self.iou_thresh)

        return coord_pred_dec_t, class_pred_nms_batch
