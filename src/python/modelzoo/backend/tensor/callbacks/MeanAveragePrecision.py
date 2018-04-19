from pprint import pprint

import numpy as np
from keras.callbacks import Callback

from modelzoo.evaluation import evaluate_generator
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import create_dirs


class MeanAveragePrecision(Callback):
    def __init__(self, predictor: Predictor, test_set: DatasetGenerator, out_file=None):
        super().__init__()

        self.out_file = out_file
        self.predictor = predictor
        self.data_gen = test_set

    def on_train_begin(self, logs={}):
        create_dirs([self.out_file])
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.predictor.net.backend = self.model
        results, labels_true, labels_pred, image_files = evaluate_generator(self.predictor,
                                                                            generator=self.data_gen,
                                                                            metrics=[MetricDetection()],
                                                                            out_file_metric=self.out_file)
        detection_result = results['MetricDetection']
        detection_result_sum = ResultByConfidence(detection_result[0])
        for d in detection_result[1:]:
            detection_result_sum = detection_result_sum + ResultByConfidence(d)

        result_mat = np.zeros((3, 11))
        for i, c in enumerate(detection_result_sum.results.keys()):
            result_mat[0, i] = detection_result_sum.results[c].precision
            result_mat[1, i] = detection_result_sum.results[c].recall
            result_mat[2, i] = c

        pprint(result_mat)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
