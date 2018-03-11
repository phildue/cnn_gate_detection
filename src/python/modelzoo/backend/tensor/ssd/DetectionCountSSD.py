from modelzoo.backend.tensor.ssd.MetricSSD import MetricSSD
import numpy as np


class DetectionCountSSD(MetricSSD):
    def __init__(self, iou_thresh=0.4, n_classes=20, iou_thresh_nms=0.4, batch_size=8,
                 conf_thresh=np.linspace(0, 1.0, 11)
                 ):
        super().__init__(iou_thresh, n_classes, iou_thresh_nms, batch_size, conf_thresh)

    def compute(self, y_true, y_pred):
        """
        Determines number of true positives, false positives, false negatives

        :return: true positives, false positives, false negatives
        """
        coord_true_t, class_true_t = self._postprocess_pred(y_true)
        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        n_tp, n_fp, n_fn = self.map_adapter.detections(coord_true_t, coord_pred_t,
                                                       class_true_t, class_pred_t,
                                                       self.conf_thresh)

        return n_tp, n_fp, n_fn
