import keras.backend as K

from modelzoo.backend.tensor.yolo.MetricYolo import MetricYolo


class PrecisionRecallYolo(MetricYolo):
    def __init__(self, n_boxes=5, grid=(13, 13), iou_thresh=0.4, n_classes=20, norm=(416, 416), iou_thresh_nms=0.4,
                 batch_size=8, confidence_levels=K.np.linspace(0, 1.0, 11)):
        super().__init__(n_boxes, grid, iou_thresh, n_classes, norm, iou_thresh_nms, batch_size, confidence_levels)

    def compute(self, y_true, y_pred):
        """
        Calculates the precision-recall for one confidence level
        :return: precision, recall
        """
        y_true = K.reshape(y_true, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])
        y_pred = K.reshape(y_pred, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])

        coord_true_t, class_true_t = self._postprocess_truth(y_true)

        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        precision, recall, n_predictions = self.map_adapter.precision_recall(coord_true_t, coord_pred_t, class_true_t,
                                                                             class_pred_t,
                                                                             self.confidence_levels)

        return precision, recall, n_predictions
