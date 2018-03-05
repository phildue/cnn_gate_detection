import keras.backend as K

from src.python.modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision


class AveragePrecisionSSD:
    def __init__(self, anchors_t, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4,
                 batch_size=8, variances=[1.0, 1.0, 1.0, 1.0]):
        self.anchors_t = anchors_t
        self.variances = variances
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.n_classes = n_classes
        self.n_boxes = self.anchors_t.shape[0]
        self.map_adapter = AveragePrecision(iou_thresh, self.n_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t):
        coord_decoded_t = coord_t
        anchor_wh = self.anchors_t[:, 2:]
        anchor_cxy = self.anchors_t[:, :2]

        coord_decoded_cx = coord_decoded_t[:, 0] * anchor_wh[:, 0] * self.variances[0]
        coord_decoded_cy = coord_decoded_t[:, 1] * anchor_wh[:, 1] * self.variances[1]
        coord_decoded_cxy = K.concatenate([coord_decoded_cx, coord_decoded_cy], -1)

        coord_decoded_cxy = coord_decoded_cxy + anchor_cxy

        coord_decoded_w = K.exp(coord_t[:, -2])
        coord_decoded_h = K.exp(coord_t[:, -1])

        coord_decoded_w = coord_decoded_w * anchor_wh[:, 0] * self.variances[2]
        coord_decoded_h = coord_decoded_h * anchor_wh[:, 1] * self.variances[3]

        coord_dec_t = K.concatenate([coord_decoded_cx, coord_decoded_w, coord_decoded_h], -1)

        return coord_dec_t

    def _filter_boxes(self, y_pred):
        coord_pred_t = y_pred[:, -4:]
        class_pred_t = y_pred[:, 1:-4]
        conf_pred_t = K.max(class_pred_t, axis=-1)

        coord_pred_dec_t = self._decode_coord(coord_pred_t)

        coord_pred_reshape_t = K.reshape(coord_pred_dec_t, (self.batch_size, -1, 4))
        conf_pred_reshape_t = K.reshape(conf_pred_t, (self.batch_size, -1, 1))

        class_pred_reshape_t = K.reshape(class_pred_t, (self.batch_size, -1, self.n_classes))

        for i in range(self.batch_size):
            idx = K.tf.image.non_max_suppression(coord_pred_reshape_t[i],
                                                 K.flatten(conf_pred_reshape_t[i]),
                                                 self.n_boxes, self.iou_thresh,
                                                 'NonMaxSuppression')
            idx = K.expand_dims(idx, 1)
            class_pred_nms = K.tf.gather_nd(class_pred_reshape_t[i], idx)
            class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(class_pred_reshape_t[0]))
            class_pred_nms = K.expand_dims(class_pred_nms, 0)
            if i == 0:
                class_pred_nms_batch = class_pred_nms
            else:
                class_pred_nms_batch = K.concatenate([class_pred_nms_batch, class_pred_nms], 0)

        return coord_pred_reshape_t, class_pred_nms_batch

    def detections(self, y_true, y_pred, conf_thresh=K.np.linspace(0, 1.0, 11)):
        """
        Determines number of true positives, false positives, false negatives

        :return: true positives, false positives, false negatives
        """

        coord_true_t, class_true_t = self._filter_boxes(y_true)

        coord_pred_t, class_pred_t = self._filter_boxes(y_pred)

        n_tp, n_fp, n_fn = self.map_adapter.detections(coord_true_t, coord_pred_t,
                                                       class_true_t, class_pred_t,
                                                       conf_thresh)

        return n_tp, n_fp, n_fn

    def precision_recall(self, y_true, y_pred, conf_thresh=K.np.linspace(0, 1.0, 11)):
        """
        Calculates the precision-recall for one confidence level
        :return: precision, recall
        """
        coord_true_t, class_true_t = self._filter_boxes(y_true)

        coord_pred_t, class_pred_t = self._filter_boxes(y_pred)

        precision, recall, n_predictions = self.map_adapter.precision_recall(coord_true_t, coord_pred_t, class_true_t,
                                                                             class_pred_t,
                                                                             conf_thresh)

        return precision, recall, n_predictions

    def average_precision(self, y_true, y_pred):
        """
        Calculates the average precision across all confidence levels

        :return: average precision
        """
        coord_true_t, class_true_t = self._filter_boxes(y_true)

        coord_pred_t, class_pred_t = self._filter_boxes(y_pred)

        average_precision = self.map_adapter.average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision
