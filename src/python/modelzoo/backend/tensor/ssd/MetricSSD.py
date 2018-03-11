from abc import abstractmethod

import keras.backend as K

from modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision
from modelzoo.backend.tensor.metrics.Metric import Metric


class MetricSSD(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, img_shape, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4,
                 batch_size=8, conf_thresh=K.np.linspace(0, 1, 11), n_boxes=9732):
        self.img_height, self.img_width, _ = img_shape
        self.conf_thresh = conf_thresh
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.map_adapter = AveragePrecision(iou_thresh, self.n_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t):
        coord_decoded_t = coord_t

        anchor_cxy = coord_t[:, -4:-2]
        anchor_wh = coord_t[:, -2:]

        variances = coord_t[:, -8:-4]

        coord_decoded_cx = coord_decoded_t[:, :, 0] * anchor_wh[:, :, 0] / variances[:, :, 0]
        coord_decoded_cx = K.expand_dims(coord_decoded_cx, -1)
        coord_decoded_cy = coord_decoded_t[:, :, 1] * anchor_wh[:, :, 1] / variances[:, :, 1]
        coord_decoded_cy = K.expand_dims(coord_decoded_cy, -1)
        coord_decoded_cxy = K.concatenate([coord_decoded_cx, coord_decoded_cy], -1)

        coord_decoded_cxy = coord_decoded_cxy + anchor_cxy
        coord_decoded_cxy *= K.constant([[[self.img_width, self.img_height]]])

        coord_decoded_w = K.exp(coord_t[:, :, -2] * variances[:, :, 2])
        coord_decoded_h = K.exp(coord_t[:, :, -1] * variances[:, :, 3])

        coord_decoded_w = coord_decoded_w * anchor_wh[:, :, 0] * self.img_width
        coord_decoded_w = K.expand_dims(coord_decoded_w, -1)

        coord_decoded_h = coord_decoded_h * anchor_wh[:, :, 1] * self.img_height
        coord_decoded_h = K.expand_dims(coord_decoded_h, -1)

        coord_dec_t = K.concatenate([coord_decoded_cxy, coord_decoded_w, coord_decoded_h], -1)

        return coord_dec_t

    def _postprocess_truth(self, label_t):
        coord_t = label_t[:, :, -13:-1]
        class_t = label_t[:, :, 1:-13]
        uniques = 1 - label_t[:, :, -1]
        coord_dec_t = self._decode_coord(coord_t)

        class_t *= K.expand_dims(uniques, -1)
        coord_dec_t *= K.expand_dims(uniques, -1)

        return coord_dec_t, class_t

    def _postprocess_pred(self, label_t):
        coord_t = label_t[:, :, -13:-1]
        class_t = K.softmax(label_t[:, :, :-4])[:, :, 1:]

        coord_dec_t = self._decode_coord(coord_t)

        class_nms_t = self._non_max_suppression(coord_dec_t, class_t)

        return coord_dec_t, class_nms_t

    def _non_max_suppression(self, coord_pred_t, class_pred_t):
        conf_pred_t = K.max(class_pred_t, axis=-1)

        class_pred_nms_batch = []
        for i in range(self.batch_size):
            idx = K.tf.image.non_max_suppression(K.cast(coord_pred_t[i], K.tf.float32),
                                                 K.cast(K.flatten(conf_pred_t[i]), K.tf.float32),
                                                 self.n_boxes, self.iou_thresh,
                                                 'NonMaxSuppression')
            idx = K.expand_dims(idx, 1)
            class_pred_nms = K.tf.gather_nd(class_pred_t[i], idx)
            class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(class_pred_t[0]))
            class_pred_nms = K.expand_dims(class_pred_nms, 0)
            class_pred_nms_batch.append(class_pred_nms)

        class_pred_nms_batch = K.concatenate(class_pred_nms_batch, 0)
        return class_pred_nms_batch
