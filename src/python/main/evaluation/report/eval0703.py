import numpy as np
from modelzoo.backend.visuals.plots.BaseHist import BaseHist

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from utils.BoundingBox import BoundingBox
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import work_dir

work_dir()


def interp(results: ResultByConfidence, recall_levels=None):
    if recall_levels is None:
        recall_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    sorted_results = results.values
    precision_raw = np.zeros((len(sorted_results, )))
    recall_raw = np.zeros((len(sorted_results, )))
    positives_raw = np.zeros((len(sorted_results, )))
    true_objects = 0
    for i, r in enumerate(sorted_results):
        precision_raw[i] = r.precision
        recall_raw[i] = r.recall
        positives_raw[i] = (r.true_positives + r.false_positives)
        true_objects = (r.true_positives + r.true_negatives)

    precision = np.zeros(shape=(len(recall_levels, )))
    positives = np.zeros(shape=(len(recall_levels, )))
    for i, r in enumerate(recall_levels):
        try:
            idx = np.where(recall_raw[:] >= r)[0][0]
            precision[i] = np.max(precision_raw[idx:])
            positives[i] = (positives_raw[idx])
        except IndexError:
            pass
    return precision, recall_levels.T, positives, true_objects


def mean_avg_prec(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    precision = np.zeros((len(detection_result), 11))
    recall = np.zeros((len(detection_result), 11))
    positives = np.zeros((len(detection_result), 11))
    true_objects = np.zeros((len(detection_result),))
    for i, result in enumerate(detection_result):
        precision[i], recall[i], positives[i], true_objects[i] = interp(result)

    mean_pr = np.mean(precision, 0)
    mean_rec = np.mean(recall, 0)
    mean_positives = np.mean(positives, 0)
    mean_true_obj = np.mean(true_objects)
    return mean_pr, mean_rec, mean_positives, mean_true_obj


def mean_prec_rec(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    precision = np.zeros((len(detection_result), 11))
    recall = np.zeros((len(detection_result), 11))
    n_predictions = np.zeros((len(detection_result), 11))
    n_objs = np.zeros((len(detection_result), 11))
    confidence = np.round(np.linspace(0, 1.0, 11), 2)

    for i, result in enumerate(detection_result):
        for j, c in enumerate(confidence):
            precision[i, j] = result.results[c].precision
            recall[i, j] = result.results[c].recall
            n_predictions[i, j] = result.results[c].true_positives + result.results[c].false_positives
            n_objs[i, j] = result.results[c].true_positives + result.results[c].false_negatives

    mean_prec = np.mean(precision, 0)
    mean_rec = np.mean(recall, 0)
    mean_pred = np.mean(n_predictions, 0)
    mean_objs = np.mean(n_objs, 0)
    return mean_prec, mean_rec, mean_pred, mean_objs, confidence


def mean_detections(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    true_positives = np.zeros((len(detection_result), 11))
    false_positives = np.zeros((len(detection_result), 11))
    n_predictions = np.zeros((len(detection_result), 11))
    n_objs = np.zeros((len(detection_result), 11))
    confidence = np.round(np.linspace(0, 1.0, 11), 2)

    for i, result in enumerate(detection_result):
        for j, c in enumerate(confidence):
            true_positives[i, j] = result.results[c].true_positives
            false_positives[i, j] = result.results[c].false_positives
            n_predictions[i, j] = result.results[c].true_positives + result.results[c].false_positives
            n_objs[i, j] = result.results[c].true_positives + result.results[c].false_negatives

    mean_tp = np.mean(true_positives, 0)
    mean_fn = np.mean(false_positives, 0)
    mean_pred = np.mean(n_predictions, 0)
    mean_objs = np.mean(n_objs, 0)
    return mean_tp, mean_fn, mean_pred, mean_objs, confidence


def mean_ap(paths):
    result_file = 'metric_result_0703.pkl'

    mean_results = []
    for rt in paths:
        results = load_file(rt + result_file)
        precision, recall, positives, true_objects = mean_avg_prec(results)
        mean_results.append((precision, recall, positives, true_objects))

    precisions = [mr[0] for mr in mean_results]
    recall = [mr[1] for mr in mean_results]
    positives = [mr[2] for mr in mean_results]
    true_objects = [mr[3] for mr in mean_results]

    return precisions, recall, positives, true_objects


def hist_prep(file_content):
    confidence_levels = np.linspace(0, 1, 11)
    positives = np.zeros((len(file_content['labels_pred']), 11))
    true_positives = np.zeros((len(file_content['labels_pred']), 11))
    ObjectLabel.classes = ['gate']
    for i, label_pred in enumerate(file_content['labels_pred']):
        for j, c in enumerate(confidence_levels, 1):
            objs = [o for o in label_pred.objects if
                    o.confidence > confidence_levels[j - 1] and o.confidence <= confidence_levels[j]]

            objs_tp = []
            bb_pred = BoundingBox.from_label(label_pred)
            bb_true = BoundingBox.from_label(file_content['labels_true'][i])

            for o_pred in bb_pred:
                if confidence_levels[j - 1] < o_pred.c <= confidence_levels[j]:
                    for o_true in bb_true:
                        if o_pred.iou(o_true) > 0.4 and o_pred.prediction == o_true.prediction:
                            objs_tp.append(o_pred)
                            break

            positives[i, j - 1] = len(objs)
            true_positives[i, j - 1] = len(objs_tp)

    mean_positives = np.mean(positives[:, 1:], 0)
    mean_tp = np.mean(true_positives[:, 1:], 0)

    return mean_positives, mean_tp, confidence_levels


fig_dir = 'doc/report/2018-07-03/fig/'
results_tiny = ['logs/tinyyolo_aligned_distort_208/0703/', 'logs/tinyyolo_10k/0703/',
                'logs/tinyyolo_aligned_distort/0703/', 'logs/tiny_20k_new/0703/', 'logs/tinyyolo_10k_new/0703/']
results_v2 = ['logs/yolov2_aligned_distort/0703/', 'logs/yolov2_10k/0703/',
              'logs/yolov2_20k_new/0703/', 'logs/yolov2_10k_new/0703/']

precisions_tiny, recall_tiny, positives_tiny, true_objects_tiny = mean_ap(results_tiny)
print('True Objects', true_objects_tiny)
print('Positives', positives_tiny)
pr_plot_tiny = BaseMultiPlot(y_data=precisions_tiny, x_data=recall_tiny,
                             y_label='Precision', x_label='Recall',
                             legend=['208:20k:aug:soft', '416:10k:hard', '416:20k:aug:soft', '416:20k:soft',
                                     '416:10k:soft'],
                             y_lim=(0.0, 1.0))

precisions_v2, recall_v2, positives_v2, true_objects_v2 = mean_ap(results_v2)

print('True Objects', true_objects_v2)
print('Positives', positives_v2)
pr_plot_v2 = BaseMultiPlot(y_data=precisions_v2, x_data=recall_v2,
                           y_label='Precision', x_label='Recall',
                           legend=['416:20k:aug:soft', '416:10k:hard', '416:20k:soft', '416:10k:soft'],
                           y_lim=(0.0, 1.0))

pr_plot_tiny.save(fig_dir + '/pr_tiny.png')
pr_plot_v2.save(fig_dir + '/pr_v2.png')

mean_prec_tiny, mean_rec_tiny, mean_pred_tiny, mean_objs_tiny, conf_tiny = mean_prec_rec(
    load_file('logs/tiny_20k_new/0703/metric_result_0703.pkl'))
mean_prec_v2, mean_rec_v2, mean_pred_v2, mean_objs_v2, conf_v2 = mean_prec_rec(
    load_file('logs/yolov2_20k_new/0703/metric_result_0703.pkl'))

prec_rec_plot_tiny = BaseMultiPlot(x_data=[conf_tiny] * 2, y_data=[mean_prec_tiny, mean_rec_tiny],
                                   x_label='Confidence', y_label='',
                                   legend=['Precision', 'Recall'],
                                   title='PrecisionRecall TinyYolo',
                                   line_style=['b--', 'g--'])
prec_rec_plot_v2 = BaseMultiPlot(x_data=[conf_v2] * 2, y_data=[mean_prec_v2, mean_rec_v2],
                                 x_label='Confidence', y_label='',
                                 legend=['Precision', 'Recall'],
                                 title='PrecisionRecall YoloV2',
                                 line_style=['b--', 'g--'])

n_pred_plot = BaseMultiPlot(x_data=[conf_v2[1:]] * 2, y_data=[mean_pred_tiny[1:], mean_pred_v2[1:]],
                            x_label='Confidence', y_label='Number Of Predictions',
                            legend=['TinyYolo', 'YoloV2'])

n_pred_plot.show(False)
n_pred_plot.save(fig_dir + '/n_pred.png')
prec_rec_plot_tiny.show(False)
prec_rec_plot_tiny.save(fig_dir + '/prec_rec_tiny.png')

prec_rec_plot_v2.show(False)
prec_rec_plot_v2.save(fig_dir + '/prec_rec_v2.png')

label_file_tiny = load_file('logs/tiny_20k_new/0703/result_0703.pkl')
positives_tiny_hist, tp_tiny_hist, conf_tiny_hist = hist_prep(label_file_tiny)
hist_tiny = BaseBarPlot(y_data=[positives_tiny_hist, tp_tiny_hist], x_data=[conf_tiny_hist[1:]] * 2, width=0.05,
                        y_label='Positives/True Positives', x_label='Confidence', title='Histogram TinyYolo',
                        colors=['blue', 'green'])
hist_tiny.show(False)
hist_tiny.save(fig_dir + '/hist_tiny.png')
label_file_v2 = load_file('logs/yolov2_20k_new/0703/result_0703.pkl')
positives_v2_hist, tp_v2_hist, conf_v2_hist = hist_prep(label_file_v2)
hist_v2 = BaseBarPlot(y_data=[positives_v2_hist, tp_v2_hist], x_data=[conf_v2_hist[1:]] * 2, width=0.05,
                      y_label='Positives/True Positives', x_label='Confidence', title='Histogram YoloV2',
                      colors=['blue', 'green'])
hist_v2.save(fig_dir + '/hist_v2.png')
hist_v2.show(True)
