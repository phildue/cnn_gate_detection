# TODO create global interface for evaluator that can be customized according to metric
from tensorflow.contrib.predictor.predictor import Predictor

from modelzoo.evaluation.ConfidenceEvaluator import ConfidenceEvaluator
from modelzoo.evaluation.Metric import Metric
from modelzoo.evaluation.MetricEvaluator import MetricEvaluator
from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import load_file


def evaluate_generator(model: Predictor, generator: DatasetGenerator, n_batches=None, metrics: [Metric] = None,
                       out_file_labels=None, out_file_metric=None,
                       verbose=True,
                       confidence_levels=11):
    if n_batches is None:
        n_batches = int(generator.n_samples / generator.batch_size)

    labels_true, labels_pred, image_files = ModelEvaluator(model, out_file_labels, verbose).evaluate_generator(
        generator, n_batches)

    if metrics:
        results = ConfidenceEvaluator(metrics, confidence_levels, out_file_metric, verbose,
                                      generator.color_format).evaluate(labels_true, labels_pred,
                                                                       image_files)
        return results, labels_true, labels_pred, image_files
    else:
        return labels_true, labels_pred, image_files


def evaluate_file(label_file: str, metrics: [Metric] = None, out_file_metric=None,
                  verbose=True,
                  color_format='bgr', confidence_levels=11):
    content = load_file(label_file)
    labels_true = content['labels_true']
    labels_pred = content['labels_pred']
    image_files = content['image_files']

    if metrics:
        results = ConfidenceEvaluator(metrics, confidence_levels, out_file_metric, verbose,
                                      color_format).evaluate(labels_true, labels_pred,
                                                             image_files)
        return results, labels_true, labels_pred, image_files
    else:
        return labels_true, labels_pred, image_files
