from abc import abstractmethod

import keras.backend as K

from modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision
from modelzoo.backend.tensor.metrics.Metric import Metric


class MetricYolo(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, n_boxes=5, grid=(13, 13), iou_thresh=0.4, n_classes=20, norm=(416, 416), iou_thresh_nms=0.4,
                 batch_size=8, confidence_levels=K.np.linspace(0, 1.0, 11)):
        self.confidence_levels = confidence_levels
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.norm = norm
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.grid = grid
        self.map_adapter = AveragePrecision(iou_thresh, grid[1] * grid[0] * n_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t):
        offset_y, offset_x = K.np.mgrid[:self.grid[0], :self.grid[1]]
        offset_y = K.constant(offset_y, K.tf.float32)
        offset_x = K.constant(offset_x, K.tf.float32)

        offset_x = K.expand_dims(offset_x, -1)
        offset_x = K.expand_dims(offset_x, 0)
        offset_x = K.tile(offset_x, (self.batch_size, 1, 1, self.n_boxes))

        offset_y = K.expand_dims(offset_y, -1)
        offset_y = K.expand_dims(offset_y, 0)
        offset_y = K.tile(offset_y, (self.batch_size, 1, 1, self.n_boxes))

        coord_t_cx = coord_t[:, :, :, :, 0] + offset_x
        coord_t_cy = coord_t[:, :, :, :, 1] + offset_y
        coord_t_cx = coord_t_cx * (self.norm[1] / self.grid[1])
        coord_t_w = coord_t[:, :, :, :, 2] * (self.norm[1] / self.grid[1])
        coord_t_cy = coord_t_cy * (self.norm[0] / self.grid[0])
        coord_t_h = coord_t[:, :, :, :, 3] * (self.norm[0] / self.grid[0])

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
        w_zero_anchors = K.np.zeros((self.batch_size, self.grid[0], self.grid[1], self.n_boxes, self.n_classes))
        w_zero_anchors[:, :, :, 0, :] = 1
        w_zero_anchors = K.constant(w_zero_anchors, dtype=K.tf.float32)

        coord_true_t = y_true[:, :, :, :, :4]
        class_true_t = y_true[:, :, :, :, 5:]
        class_true_t = w_zero_anchors * class_true_t

        coord_true_dec_t = self._decode_coord(coord_true_t)

        coord_true_reshape_t = K.reshape(coord_true_dec_t, (self.batch_size, -1, 4))
        class_true_reshape_t = K.reshape(class_true_t, (self.batch_size, -1, 1))

        return coord_true_reshape_t, class_true_reshape_t

    def _postprocess_pred(self, y_pred):
        coord_pred_t = y_pred[:, :, :, :, :4]
        class_pred_t = y_pred[:, :, :, :, 5:]
        conf_pred_t = y_pred[:, :, :, :, 4]

        coord_pred_dec_t = self._decode_coord(coord_pred_t)

        coord_pred_reshape_t = K.reshape(coord_pred_dec_t, (self.batch_size, -1, 4))
        conf_pred_reshape_t = K.reshape(conf_pred_t, (self.batch_size, -1, 1))

        class_pred_reshape_t = K.reshape(class_pred_t, (self.batch_size, -1, self.n_classes))

        class_pred_nms_batch = self.map_adapter.non_max_suppression_batch(coord_pred_t,
                                                                          class_pred_reshape_t,
                                                                          self.batch_size,
                                                                          self.n_boxes * self.grid[0] * self.grid[1],
                                                                          self.iou_thresh)

        return coord_pred_reshape_t, class_pred_nms_batch
