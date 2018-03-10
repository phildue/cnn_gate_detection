from abc import abstractmethod

import keras.backend as K

from modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision
from modelzoo.backend.tensor.metrics.Metric import Metric


class MetricSSD(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, anchors_t, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4,
                 batch_size=8, variances=[1.0, 1.0, 1.0, 1.0], conf_thresh=K.np.linspace(0, 1, 11)):
        self.conf_thresh = conf_thresh
        self.anchors_t = anchors_t
        self.variances = variances
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.n_classes = n_classes
        self.n_boxes = self.anchors_t.shape[0]
        self.map_adapter = AveragePrecision(iou_thresh, self.n_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t):
        coord_decoded_t = coord_t
        anchor_wh = K.expand_dims(self.anchors_t[:, 2:], 0)
        anchor_cxy = K.expand_dims(self.anchors_t[:, :2], 0)

        coord_decoded_cx = coord_decoded_t[:, :, 0] * anchor_wh[:, :, 0] * self.variances[0]
        coord_decoded_cx = K.expand_dims(coord_decoded_cx, -1)
        coord_decoded_cy = coord_decoded_t[:, :, 1] * anchor_wh[:, :, 1] * self.variances[1]
        coord_decoded_cy = K.expand_dims(coord_decoded_cy, -1)
        coord_decoded_cxy = K.concatenate([coord_decoded_cx, coord_decoded_cy], -1)

        coord_decoded_cxy = coord_decoded_cxy + anchor_cxy

        coord_decoded_w = K.exp(coord_t[:, :, -2])
        coord_decoded_h = K.exp(coord_t[:, :, -1])

        coord_decoded_w = coord_decoded_w * anchor_wh[:, :, 0] * self.variances[2]
        coord_decoded_w = K.expand_dims(coord_decoded_w, -1)

        coord_decoded_h = coord_decoded_h * anchor_wh[:, :, 1] * self.variances[3]
        coord_decoded_h = K.expand_dims(coord_decoded_h, -1)

        coord_dec_t = K.concatenate([coord_decoded_cxy, coord_decoded_w, coord_decoded_h], -1)

        return coord_dec_t

    def _filter_boxes(self, label_t):
        coord_pred_t = label_t[:, :, -4:]
        class_pred_t = label_t[:, :, 1:-4]
        conf_pred_t = K.max(class_pred_t, axis=-1)

        coord_pred_dec_t = self._decode_coord(coord_pred_t)

        coord_pred_reshape_t = K.reshape(coord_pred_dec_t, (self.batch_size, -1, 4))
        conf_pred_reshape_t = K.reshape(conf_pred_t, (self.batch_size, -1, 1))

        class_pred_nms_batch = []
        for i in range(self.batch_size):
            idx = K.tf.image.non_max_suppression(K.cast(coord_pred_reshape_t[i], K.tf.float32),
                                                 K.cast(K.flatten(conf_pred_reshape_t[i]), K.tf.float32),
                                                 self.n_boxes, self.iou_thresh,
                                                 'NonMaxSuppression')
            idx = K.expand_dims(idx, 1)
            class_pred_nms = K.tf.gather_nd(class_pred_t[i], idx)
            class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(class_pred_t[0]))
            class_pred_nms = K.expand_dims(class_pred_nms, 0)
            class_pred_nms_batch.append(class_pred_nms)

        class_pred_nms_batch = K.concatenate(class_pred_nms_batch, 0)
        return coord_pred_reshape_t, class_pred_nms_batch
