from modelzoo.models.ssd import MetricSSD


class AveragePrecisionSSD(MetricSSD):


    def compute(self, y_true, y_pred):
        """
       Calculates the average precision across all confidence levels

       :return: average precision
       """
        coord_true_t, class_true_t = self._postprocess_truth(y_true)
        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        average_precision = self.map_adapter.mean_average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision
