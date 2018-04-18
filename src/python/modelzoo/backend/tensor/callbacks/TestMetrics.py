from keras.callbacks import Callback

from modelzoo.evaluation.MetricEvaluator import DataEvaluator
from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from utils.fileaccess.DatasetGenerator import DatasetGenerator


class TestMetric(Callback):
    def __init__(self, test_set: DatasetGenerator, evaluator: ModelEvaluator, file_evaluator: DataEvaluator):
        super().__init__()
        self.file_evaluator = file_evaluator
        self.evaluator = evaluator
        self.out_file = file_evaluator.out_file
        self.data_gen = test_set

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.evaluator.model.net.backend = self.model
        self.file_evaluator.out_file = self.out_file + "-{0:000d}.pkl".format(epoch)
        labels_true, labels_pred, image_files = self.evaluator.evaluate_generator(self.data_gen)
        self.file_evaluator.evaluate(labels_true, labels_pred, image_files)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
