import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
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


def avg_pr_file(src_file):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_pr, mean_rec = average_precision_recall(detection_result)
    return mean_pr, mean_rec


def pr_plot(files, legend, title, line_style=None, y_range=(0.5, 1.0)):
    recalls = []
    precisions = []
    for f in files:
        mean_p, mean_r = avg_pr_file(f)
        recalls.append(mean_r)
        precisions.append(mean_p)

    return BaseMultiPlot(x_data=recalls,
                         y_data=precisions,
                         y_label='Precision',
                         x_label='Recall',
                         y_lim=y_range,
                         legend=legend,
                         title=title,
                         line_style=line_style)


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


def detection_plot(src_file, title):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_tp, mean_fp, mean_fn, mean_pred, mean_objs, confidence = mean_detections(detection_result)
    return BaseMultiPlot(x_data=[confidence] * 5, y_data=[mean_tp, mean_pred, mean_objs],
                         legend=['True Positives', 'Predictions', 'True Objects'],
                         x_label='Confidence',
                         y_label='Predictions',
                         line_style=['.-', '.-', ':'],
                         title=title)


cd_work()

pr_basement_all = pr_plot(files=['logs/gatev5_industrial/results/industrial--024.pkl',
                                 'logs/gatev5_daylight/results/industrial--024.pkl',
                                 'logs/gatev5_mixed/results/industrial--024.pkl',
                                 'logs/tiny_industrial/results/industrial--024.pkl',
                                 'logs/tiny_daylight/results/industrial--024.pkl',
                                 'logs/tiny_mixed/results/industrial--024.pkl',
                                 'logs/v2_industrial/results/industrial--019.pkl',
                                 'logs/v2_daylight/results/industrial--023.pkl',
                                 'logs/v2_mixed/results/industrial--019.pkl'
                                 ],
                          legend=['GateNet-Basement',
                                  'GateNet-Daylight',
                                  'GateNet-Mixed',
                                  'GateNet6-Mixed',
                                  'Tiny-Basement',
                                  'Tiny-Daylight',
                                  'Tiny-Mixed',
                                  'V2-Basement',
                                  'V2-Daylight',
                                  'V2-Mixed'
                                  ],
                          title='Test on Basement',
                          line_style=['x-', 'x-', 'x-', 'x-', 'o:', 'o:', 'o:', '.-', '.-', '.-'])
pr_daylight_all = pr_plot(files=['logs/gatev5_industrial/results/daylight--024.pkl',
                                 'logs/gatev5_daylight/results/daylight--024.pkl',
                                 'logs/gatev5_mixed/results/daylight--024.pkl',
                                 'logs/tiny_industrial/results/daylight--024.pkl',
                                 'logs/tiny_daylight/results/daylight--023.pkl',
                                 'logs/tiny_mixed/results/daylight--024.pkl',
                                 'logs/v2_industrial/results/daylight--019.pkl',
                                 'logs/v2_daylight/results/daylight--023.pkl',
                                 'logs/v2_mixed/results/daylight--019.pkl'
                                 ],
                          legend=['GateNet-Basement',
                                  'GateNet-Daylight',
                                  'GateNet-Mixed',
                                  'GateNet6-Mixed',
                                  'Tiny-Basement',
                                  'Tiny-Daylight',
                                  'Tiny-Mixed',
                                  'V2-Basement',
                                  'V2-Daylight',
                                  'V2-Mixed'
                                  ],
                          title='Test on Daylight',
                          line_style=['x-', 'x-', 'x-', 'x-', 'o:', 'o:', 'o:', '.-', '.-', '.-'])

pr_daylight_tuning = pr_plot(files=[
    'logs/gatev5_mixed/results/daylight--024.pkl',
    'logs/gatev6-1/results/daylight--017.pkl',
    'logs/gatev7_mixed/results/daylight--022.pkl',
    'logs/gatev8_mixed/results/daylight--020.pkl',
    'logs/tiny_mixed/results/daylight--023.pkl',
    'logs/v2_mixed/results/daylight--019.pkl'
],
    legend=['GateNet5-Mixed',
            'GateNet6-Mixed',
            'GateNet7-Mixed',
            'GateNet8-Mixed',
            'Tiny-Mixed',
            'V2-Mixed'
            ],
    title='Test on Daylight',
    line_style=['x-', 'x-', 'x-', 'x-', 'o:', '.-', ],
    y_range=(0.8, 1.0))

pr_industry_tuning = pr_plot(files=[
    'logs/gatev5_mixed/results/industrial--024.pkl',
    'logs/gatev6-1/results/industrial--017.pkl',
    'logs/gatev7_mixed/results/industrial--022.pkl',
    'logs/gatev8_mixed/results/industrial--020.pkl',
    'logs/tiny_mixed/results/industrial--024.pkl',
    'logs/v2_mixed/results/industrial--019.pkl'
],
    legend=['GateNet5-Mixed',
            'GateNet6-Mixed',
            'GateNet7-Mixed',
            'GateNet8-Mixed',
            'Tiny-Mixed',
            'V2-Mixed'
            ],
    title='Test on Industry',
    line_style=['x-', 'x-', 'x-','x-', 'o:', '.-', ],
    y_range=(0.8, 1.0))

detection_gate = detection_plot('logs/gatev5_mixed/results/daylight--024.pkl', 'GateNet on Daylight')
detection_gate7 = detection_plot('logs/gatev7_mixed/results/daylight--009.pkl', 'GateNet7 on Daylight')
detection_gate6 = detection_plot('logs/gatev6-1/results/daylight--009.pkl', 'GateNet6 on Daylight')
detection_v2 = detection_plot('logs/v2_mixed/results/daylight--019.pkl', 'YoloV2 on Daylight')
detection_tiny = detection_plot('logs/tiny_mixed/results/daylight--023.pkl', 'TinyYolo on Daylight')

# pr_daylight_all.show(False)
# pr_basement_all.show(False)
# pr_daylight_tuning.show(False)
# detection_gate7.show(False)
# detection_gate6.show(False)
# pr_industry_tuning.show(True)

# detection_gate6.show(False)
# detection_gate.show(False)
# detection_v2.show(False)
# detection_tiny.show()

# pr_basement_all.save('doc/report/2018-04-22/fig/pr_basement_all.png')
# pr_daylight_all.save('doc/report/2018-04-22/fig/pr_daylight_all.png')
# detection_gate.save('doc/report/2018-04-22/fig/detection_gate.png')
# detection_v2.save('doc/report/2018-04-22/fig/detection_v2.png')
# detection_tiny.save('doc/report/2018-04-22/fig/detection_tiny.png')
pr_daylight_tuning.save('doc/report/2018-04-22/fig/pr_daylight_tuning.png')
pr_industry_tuning.save('doc/report/2018-04-22/fig/pr_industry_tuning.png')
# detection_gate7.save('doc/report/2018-04-22/fig/detection7_tuning.png')
# detection_gate6.save('doc/report/2018-04-22/fig/detection6_tuning.png')