import keras.backend as K

from modelzoo.backend.tensor.ssd.MetricSSD import MetricSSD


class AveragePrecisionSSD(MetricSSD):
    def __init__(self, anchors_t, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4, batch_size=8,
                 variances=[1.0, 1.0, 1.0, 1.0]):
        super().__init__(anchors_t, iou_thresh, n_classes, iou_thresh_nms, batch_size, variances)

    def compute(self, y_true, y_pred):
        """
       Calculates the average precision across all confidence levels

       :return: average precision
       """
        coord_true_t, class_true_t = self._filter_boxes(y_true)
        class_pred_t = K.softmax(y_pred[:, :, :-4])
        coord_pred_t = y_pred[:, :, -4:]
        y_pred_pp = K.concatenate([class_pred_t, coord_pred_t], -1)
        coord_pred_t, class_pred_t = self._filter_boxes(y_pred_pp)

        average_precision = self.map_adapter.average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision
