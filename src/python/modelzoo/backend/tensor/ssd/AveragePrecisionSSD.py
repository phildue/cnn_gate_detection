from modelzoo.backend.tensor.ssd.MetricSSD import MetricSSD


class AveragePrecisionSSD(MetricSSD):
    def __init__(self, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4, batch_size=8):
        super().__init__(iou_thresh, n_classes, iou_thresh_nms, batch_size)

    def compute(self, y_true, y_pred):
        """
       Calculates the average precision across all confidence levels

       :return: average precision
       """
        coord_true_t, class_true_t = self._postprocess_pred(y_true)
        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        average_precision = self.map_adapter.average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision
