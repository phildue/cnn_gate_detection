from modelzoo.evaluation.MetricDetection import DetectionEvaluator
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence

from modelzoo.callbacks.Evaluator import Evaluator
from modelzoo.evaluation import evaluate_generator
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator


class MeanAveragePrecision(Evaluator):
    def __init__(self, predictor: Predictor, test_set: DatasetGenerator, out_file=None, period=2):
        super().__init__(predictor, test_set, [DetectionEvaluator()], out_file, period)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            self.predictor.net.backend = self.model
            results, labels_true, labels_pred, image_files = evaluate_generator(self.predictor,
                                                                                generator=self.data_gen,
                                                                                metrics=self.metrics,
                                                                                out_file_metric=self.out_file + '{0:03d}'.format(
                                                                                    epoch))
            detection_result = results['MetricDetection']
            mean_average_precision = 0
            for d in detection_result:
                mean_average_precision += 1 / len(detection_result) * ResultByConfidence(d).average_precision

            print("Map: {0:0.2f}".format(mean_average_precision))
            return
