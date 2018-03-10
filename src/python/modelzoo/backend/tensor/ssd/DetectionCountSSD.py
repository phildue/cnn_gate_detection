import keras.backend as K

from modelzoo.backend.tensor.ssd.MetricSSD import MetricSSD


class DetectionCountSSD(MetricSSD):
    def __init__(self, anchors_t, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4, batch_size=8,
                 variances=[1.0, 1.0, 1.0, 1.0], conf_thresh=K.np.linspace(0, 1.0, 11)
                 ):
        super().__init__(anchors_t, iou_thresh, n_classes, iou_thresh_nms, batch_size, variances, conf_thresh)

    def compute(self, y_true, y_pred):
        """
        Determines number of true positives, false positives, false negatives

        :return: true positives, false positives, false negatives
        """
        coord_true_t, class_true_t = self._filter_boxes(y_true)
        class_pred_t = K.softmax(y_pred[:, :, :-4])
        coord_pred_t = y_pred[:, :, -4:]
        y_pred_pp = K.concatenate([class_pred_t, coord_pred_t], -1)
        coord_pred_t, class_pred_t = self._filter_boxes(y_pred_pp)

        n_tp, n_fp, n_fn = self.map_adapter.detections(coord_true_t, coord_pred_t,
                                                       class_true_t, class_pred_t,
                                                       self.conf_thresh)

        return n_tp, n_fp, n_fn
