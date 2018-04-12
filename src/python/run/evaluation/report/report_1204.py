import numpy as np

from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.BoundingBox import BoundingBox
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work


def mean_detections(result_by_conf):
    true_positives = np.zeros((len(result_by_conf), 10))
    false_positives = np.zeros((len(result_by_conf), 10))
    false_negatives = np.zeros((len(result_by_conf), 10))
    n_predictions = np.zeros((len(result_by_conf), 10))
    n_objs = np.zeros((len(result_by_conf), 10))
    confidence = np.round(np.linspace(0.1, 1.0, 10), 2)

    for i, result in enumerate(result_by_conf):
        for j, c in enumerate(confidence):
            true_positives[i, j] = result.results[c].true_positives
            false_positives[i, j] = result.results[c].false_positives
            false_negatives[i, j] = result.results[c].false_negatives
            n_predictions[i, j] = result.results[c].true_positives + result.results[c].false_positives
            n_objs[i, j] = result.results[c].true_positives + result.results[c].false_negatives

    mean_tp = np.mean(true_positives, 0)
    mean_fp = np.mean(false_positives, 0)
    mean_fn = np.mean(false_negatives, 0)
    mean_pred = np.mean(n_predictions, 0)
    mean_objs = np.mean(n_objs, 0)
    return mean_tp, mean_fp, mean_fn, mean_pred, mean_objs, confidence

def mean_pr_c(result_by_conf):
    precision = np.zeros((len(result_by_conf), 10))
    recall = np.zeros((len(result_by_conf), 10))
    confidence = np.round(np.linspace(0.1, 1.0, 10), 2)

    for i, result in enumerate(result_by_conf):
        for j, c in enumerate(confidence):
            precision[i, j] = result.results[c].precision
            recall[i, j] = result.results[c].recall

    mean_precision = np.mean(precision, 0)
    mean_recall = np.mean(recall, 0)
    return mean_precision, mean_recall, confidence

cd_work()

result_path = 'logs/tiny_industrial/results/'
result_file = 'industrial_room--199.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_tp_tiny, mean_fp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny, confidence = mean_detections(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence, confidence, confidence],
                                    y_data=[mean_tp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny],
                                    y_label='Number of Bounding Boxes',
                                    line_style=['-', '-', '-', 'y--'],
                                    x_label='Confidence',
                                    y_lim=(0, 3.0),
                                    legend=['True Positives', 'False Negatives', 'Total Predictions',
                                            'True Objects'],
                                    title='Tiny Yolo without Cats').show(False)

result_path = 'logs/tiny_industrial_cats/results/'
result_file = 'industrial_room--199.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_tp_tiny, mean_fp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny, confidence = mean_detections(detection_result)
tiny_cats_detection_plot = BaseMultiPlot(x_data=[confidence, confidence, confidence, confidence],
                                         y_data=[mean_tp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny],
                                         line_style=['-','-','-','y--'],
                                         y_label='Number of Bounding Boxes',
                                         x_label='Confidence',
                                         y_lim=(0, 3.0),
                                         legend=['True Positives', 'False Negatives', 'Total Predictions',
                                                 'True Objects'],
                                         title='Tiny Yolo with Cats').show(False)


result_path = 'logs/tiny_industrial/results/'
result_file = 'industrial_room--199.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
precision,recall, confidence = mean_pr_c(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence],
                                    y_data=[precision,recall],
                                    y_label='',
                                    x_label='Confidence',
                                    legend=['Precision', 'Recall'],
                                    y_lim=(0,1.0),
                                    title='Tiny Yolo without Cats').show(False)

result_path = 'logs/tiny_industrial_cats/results/'
result_file = 'industrial_room--199.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
precision,recall, confidence = mean_pr_c(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence],
                                    y_data=[precision,recall],
                                    y_label='',
                                    x_label='Confidence',
                                    legend=['Precision', 'Recall'],
                                    y_lim=(0, 1.0),
                                    title='Tiny Yolo with Cats').show(False)


result_path = 'logs/v2_industrial/results/'
result_file = 'industrial_room--150.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_tp_tiny, mean_fp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny, confidence = mean_detections(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence, confidence, confidence],
                                    y_data=[mean_tp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny],
                                    y_label='Number of Bounding Boxes',
                                    line_style=['-', '-', '-', 'y--'],
                                    x_label='Confidence',
                                    y_lim=(0, 3.0),
                                    legend=['True Positives', 'False Negatives', 'Total Predictions',
                                            'True Objects'],
                                    title='YoloV2 without Cats').show(False)

result_path = 'logs/v2_industrial_cats/results/'
result_file = 'industrial_room--150.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_tp_tiny, mean_fp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny, confidence = mean_detections(detection_result)
tiny_cats_detection_plot = BaseMultiPlot(x_data=[confidence, confidence, confidence, confidence],
                                         y_data=[mean_tp_tiny, mean_fn_tiny, mean_pred, mean_obs_tiny],
                                         line_style=['-','-','-','y--'],
                                         y_label='Number of Bounding Boxes',
                                         x_label='Confidence',
                                         y_lim=(0, 3.0),
                                         legend=['True Positives', 'False Negatives', 'Total Predictions',
                                                 'True Objects'],
                                         title='YoloV2 with Cats').show(False)


result_path = 'logs/v2_industrial/results/'
result_file = 'industrial_room--150.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
precision,recall, confidence = mean_pr_c(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence],
                                    y_data=[precision,recall],
                                    y_label='',
                                    x_label='Confidence',
                                    legend=['Precision', 'Recall'],
                                    y_lim=(0,1.0),
                                    title='YoloV2 without Cats').show(False)

result_path = 'logs/v2_industrial_cats/results/'
result_file = 'industrial_room--150.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
precision,recall, confidence = mean_pr_c(detection_result)
tiny_detection_plot = BaseMultiPlot(x_data=[confidence, confidence],
                                    y_data=[precision,recall],
                                    y_label='',
                                    x_label='Confidence',
                                    legend=['Precision', 'Recall'],
                                    y_lim=(0, 1.0),
                                    title='YoloV2 with Cats').show(True)
# result_path = 'logs/tiny_industrial/results/'
# result_file = 'industrial_room--199.pkl'
# results = load_file(result_path + result_file)
# detection_result = results['results']['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# mean_pr_tiny, mean_rec_tiny = average_precision_recall(detection_result)
#
# result_path = 'logs/tiny_industrial_cats/results/'
# result_file = 'industrial_room--199.pkl'
# results = load_file(result_path + result_file)
# detection_result = results['results']['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# mean_pr_tiny_cats, mean_rec_tiny_cats = average_precision_recall(detection_result)
#
# BaseMultiPlot(y_data=[mean_pr_tiny_cats, mean_pr_tiny],
#               x_data=[mean_rec_tiny_cats, mean_rec_tiny],
#               y_label='Precision', x_label='Recall',
#               legend=['Cats', 'Gates'],
#               title='Tiny Yolo on Industrial Room').show(False)
#
# result_path = 'logs/v2_industrial/results/'
# result_file = 'industrial_room--150.pkl'
# results = load_file(result_path + result_file)
# detection_result = results['results']['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# mean_pr_tiny, mean_rec_tiny = average_precision_recall(detection_result)
#
# result_path = 'logs/v2_industrial_cats/results/'
# result_file = 'industrial_room--150.pkl'
# results = load_file(result_path + result_file)
# detection_result = results['results']['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# mean_pr_tiny_cats, mean_rec_tiny_cats = average_precision_recall(detection_result)
#
# BaseMultiPlot(y_data=[mean_pr_tiny_cats, mean_pr_tiny],
#               x_data=[mean_rec_tiny_cats, mean_rec_tiny],
#               y_label='Precision', x_label='Recall',
#               legend=['Cats', 'Gates'],
#               title='YoloV2 on Industrial Room').show()
