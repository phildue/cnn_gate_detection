import numpy as np

from src.python.modelzoo.backend.visuals import BaseMultiPlot
from src.python.modelzoo.evaluation import DetectionResult
from src.python.utils.fileaccess import load_file


class StreamAnalyzer:
    def __init__(self, result_file):
        experiments = load_file(result_file)
        self.metric_results = experiments['results']['MetricOneGate']
        self.labels_true = experiments['labels_true']
        self.labels_pred = experiments['labels_pred']

    def loc_error_plot(self):
        loc_error = []
        dist = []
        confidences = []

        for i in range(len(self.labels_true)):
            if self.labels_pred[i].objects and \
                    self.labels_true[i].objects and \
                            self.metric_results[i][1] is not None:
                distance = self.metric_results[i][1]
                loc_error.append(np.mean(distance))
                p = self.labels_true[i].objects[0].position
                dist.append(np.array([np.sqrt(p.dist_forward ** 2 + p.dist_side ** 2 + p.lift ** 2)]))
                confidence = self.labels_pred[i].objects[0].confidence
                confidences.append(np.array(confidence) * 100)

        loc_error = np.vstack(loc_error)
        dist = np.vstack(dist)
        confidences = np.vstack(confidences)
        loc_error_dist = np.vstack([dist.T, loc_error.T, confidences.T]).T
        loc_error_dist = loc_error_dist[loc_error_dist[:, 0].argsort()]

        return BaseMultiPlot(x_data=[loc_error_dist[:, 0], loc_error_dist[:, 0]],
                             y_data=[loc_error_dist[:, 1], loc_error_dist[:, 2]],
                             title='Localization Error/Confidence',
                             y_label='', x_label='Distance To Gate Center', y_lim=(0, 150))

    def detection_eval(self):
        detection_results_sum = DetectionResult(0, 0, 0, 0)
        for r in self.detection_results:
            detection_results_sum += r

        print(detection_results_sum)

        print("Precision:" + str(detection_results_sum.precision))
        print("True Positive Rate:" + str(detection_results_sum.recall))
        print("False Positive Rate:" + str(detection_results_sum.fp_rate))
