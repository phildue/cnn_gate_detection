from keras.callbacks import Callback
from tensorflow.contrib.tpu.profiler.op_profile_pb2 import Metrics

from modelzoo.evaluation import evaluate_generator
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import create_dirs


class Evaluator(Callback):
    def __init__(self, predictor: Predictor, test_set: DatasetGenerator, metrics: [Metrics], out_file=None):
        super().__init__()

        self.out_file = out_file
        self.predictor = predictor
        self.metrics = metrics
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
        evaluate_generator(self.predictor,
                           generator=self.data_gen,
                           metrics=self.metrics,
                           out_file_metric=self.out_file)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
