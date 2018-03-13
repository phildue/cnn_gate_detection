import keras.backend as K

from modelzoo.backend.tensor.ssd.MetricSSD import MetricSSD


class PrecisionRecallSSD(MetricSSD):
    def __init__(self, iou_thresh_match=0.4, n_classes=20, iou_thresh_nms=0.4, batch_size=8,
                 conf_thresh=K.np.linspace(0, 1, 11)):
        super().__init__(iou_thresh, n_classes, iou_thresh_nms, batch_size, conf_thresh)

    def compute(self, y_true, y_pred):
        """
        Calculates the precision-recall for one confidence level
        :return: precision, recall
        """
        coord_true_t, class_true_t = self._postprocess_pred(y_true)
        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        precision, recall, n_predictions = self.map_adapter.precision_recall(coord_true_t, coord_pred_t, class_true_t,
                                                                             class_pred_t,
                                                                             self.conf_thresh)

        return precision, recall, n_predictions
