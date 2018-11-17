import keras.backend as K

from modelzoo.models.yolo import MetricYolo


class AveragePrecisionYolo(MetricYolo):
    def __init__(self, n_boxes, grid, iou_thresh, n_classes, norm, iou_thresh_nms,
                 batch_size, confidence_levels=K.np.linspace(0, 1.0, 11)):
        super().__init__(n_boxes, grid, iou_thresh, n_classes, norm, iou_thresh_nms, batch_size, confidence_levels)

    def compute(self, y_true, y_pred):
        """
        Calculates the average precision across all confidence levels

        :return: average precision
        """

        conf_true_t, class_true_t, coord_true_t = self._postprocess_truth(y_true)

        conf_pred_t, class_pred_t, coord_pred_t = self._postprocess_pred(y_pred)

        average_precision = self.map_adapter.mean_average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision
