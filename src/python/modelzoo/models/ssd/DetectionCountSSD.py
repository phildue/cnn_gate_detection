from modelzoo.models.ssd import MetricSSD


class DetectionCountSSD(MetricSSD):
    def __init__(self, iou_thresh_match=0.4, iou_thresh_nms=0.4, batch_size=8):
        super().__init__(iou_thresh_nms=iou_thresh_nms,
                         iou_thresh_match=iou_thresh_match,
                         batch_size=batch_size)

    def compute(self, y_true, y_pred):
        """
        Determines number of true positives, false positives, false negatives

        :return: true positives, false positives, false negatives
        """
        coord_true_t, class_true_t = self._postprocess_truth(y_true)
        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        n_tp, n_fp, n_fn = self.map_adapter.detections(coord_true_t, coord_pred_t,
                                                       class_true_t, class_pred_t,
                                                       self.conf_thresh)

        return n_tp, n_fp, n_fn
