import numpy as np

from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work


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


def speed_plot(src_files, names):
    curves = []
    ind = []
    names_lookup = []
    for i, src_file in enumerate(src_files, 1):
        results = load_file(src_file)
        results_pred = results['results_pred']
        results_pp = results['results_pp']
        results_total = np.sum([results_pred, results_pp], 0)
        c = [np.mean(results_pred)]
        curves.extend(c)
        ind.extend([i] * len(c))
        names_lookup.extend([names[i - 1]] * len(c))
    return BaseBarPlot(x_data=ind,
                       names=names_lookup,
                       y_data=curves,
                       colors=None,
                       width=0.4)


def performance_speed_plot(performance_files, speed_files, names):
    times = []
    performances = []
    symbols = ['x', 'o', '*']
    linestyle = []
    for i in range(len(performance_files)):
        speed_file = speed_files[i]
        performance_file = performance_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        times.append([t_pred])

        performance_file_cont = load_file(performance_file)
        detection_result = performance_file_cont['results']['MetricDetection']
        detection_result = [ResultByConfidence(d) for d in detection_result]
        mean_pr, mean_r = average_precision_recall(detection_result)
        performances.append([np.mean(mean_pr)])
        linestyle.append(symbols[i % len(symbols)])

    return BaseMultiPlot(x_data=times, x_label='Inference Time [s]',
                         y_data=performances, y_label='Mean Average Precision',
                         line_style=linestyle,
                         legend=names)


def params_speed_plot(params, speed_files, names):
    times = []
    performances = []
    for i in range(len(speed_files)):
        speed_file = speed_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        times.append([t_pred])

        performances.append([params[i]])

    return BaseMultiPlot(x_data=times, x_label='Inference Time [s]',
                         y_data=performances, y_label='Weights',
                         line_style=['o'] * len(speed_files),
                         legend=names)


def layers_speed_plot(n_layers, speed_files, names):
    times = []
    performances = []
    for i in range(len(speed_files)):
        speed_file = speed_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        print(np.std(speed_file_cont['results_pred'][1:]))
        times.append([t_pred])

        performances.append([n_layers[i]])

    return BaseMultiPlot(x_data=times, x_label='Inference Time [s]',
                         y_data=performances, y_label='Layers',
                         line_style=['o'] * len(speed_files),
                         legend=names)


cd_work()

pr_daylight_tuning = pr_plot(files=[
    'out/gatev5_mixed/results/daylight--024.pkl',
    'out/gatev6-1/results/daylight--017.pkl',
    'out/gatev7_mixed/results/daylight--022.pkl',
    'out/gatev8_mixed/results/daylight--020.pkl',
    'out/gatev9_mixed/results/daylight--020.pkl',
    'out/gatev10_mixed/results/daylight--020.pkl',
    'out/gatev11_mixed/results/daylight--018.pkl',
    'out/gatev12_mixed/results/daylight--018.pkl',
    'out/gatev13_mixed/results/daylight--018.pkl',
    'out/gatev14_mixed/results/daylight--018.pkl',
    'out/gatev15_mixed/results/test_2--009.pkl',
    'out/gatev16_mixed/results/test_2--019.pkl',
    'out/gatev17_mixed/results/test_2--019.pkl',
    'out/tiny_mixed/results/daylight--023.pkl',
    'out/v2_mixed/results/daylight--019.pkl'
],
    legend=['GateNet5-Mixed',
            'GateNet6-Mixed',
            'GateNet7-Mixed',
            'GateNet8-Mixed',
            'GateNet9-Mixed',
            'GateNet10-Mixed',
            'GateNet11-Mixed',
            'GateNet12-Mixed',
            'GateNet13-Mixed',
            'GateNet14-Mixed',
            'GateNet15-Mixed',
            'GateNet16-Mixed',
            'GateNet17-Mixed',
            'Tiny-Mixed',
            'V2-Mixed'
            ],
    title='Test on Daylight',
    line_style=['x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'o:', '.-', ],
    y_range=(0.8, 1.0))
#
pr_basement_tuning = pr_plot(files=[
    'out/gatev5_mixed/results/industrial--023.pkl',
    'out/gatev6-1/results/industrial--017.pkl',
    'out/gatev7_mixed/results/industrial--022.pkl',
    'out/gatev8_mixed/results/industrial--020.pkl',
    'out/gatev9_mixed/results/industrial--020.pkl',
    'out/gatev10_mixed/results/industrial--020.pkl',
    'out/gatev11_mixed/results/industrial--018.pkl',
    'out/gatev12_mixed/results/industrial--018.pkl',
    'out/gatev13_mixed/results/industrial--018.pkl',
    'out/gatev14_mixed/results/industrial--018.pkl',
    'out/gatev15_mixed/results/test_1--009.pkl',
    'out/gatev16_mixed/results/test_1--019.pkl',
    'out/gatev17_mixed/results/test_1--019.pkl',
    'out/tiny_mixed/results/industrial--023.pkl',
    'out/v2_mixed/results/industrial--019.pkl'
],
    legend=['GateNet5-Mixed',
            'GateNet6-Mixed',
            'GateNet7-Mixed',
            'GateNet8-Mixed',
            'GateNet9-Mixed',
            'GateNet10-Mixed',
            'GateNet11-Mixed',
            'GateNet12-Mixed',
            'GateNet13-Mixed',
            'GateNet14-Mixed',
            'GateNet15-Mixed',
            'GateNet16-Mixed',
            'GateNet17-Mixed',
            'Tiny-Mixed',
            'V2-Mixed'
            ],
    title='Test on Basement',
    line_style=['x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'o:', '.-', ],
    y_range=(0.8, 1.0))
#
ps_plot = performance_speed_plot(performance_files=[
    'out/gatev5_mixed/results/daylight--023.pkl',
    'out/gatev8_mixed/results/daylight--020.pkl',
    'out/gatev9_mixed/results/daylight--020.pkl',
    'out/gatev10_mixed/results/daylight--020.pkl',
    'out/gatev11_mixed/results/daylight--018.pkl',
    'out/gatev12_mixed/results/daylight--018.pkl',
    'out/gatev13_mixed/results/daylight--018.pkl',
    'out/gatev14_mixed/results/daylight--018.pkl',
    'out/gatev15_mixed/results/test_2--009.pkl',
    'out/gatev16_mixed/results/test_2--018.pkl',
    'out/gatev17_mixed/results/test_2--018.pkl',
    'out/tiny_mixed/results/industrial--023.pkl',
    'out/v2_mixed/results/industrial--019.pkl'
],
    speed_files=['out/gatev5_mixed/speed/result.pkl',
                 'out/gatev8_mixed/speed/result.pkl',
                 'out/gatev9_mixed/speed/result.pkl',
                 'out/gatev10_mixed/speed/result.pkl',
                 'out/gatev11_mixed/speed/result.pkl',
                 'out/gatev12_mixed/speed/result.pkl',
                 'out/gatev13_mixed/speed/result.pkl',
                 'out/gatev14_mixed/speed/speed_result.pkl',
                 'out/gatev15_mixed/speed/result.pkl',
                 'out/gatev16_mixed/speed/speed_result.pkl',
                 'out/gatev17_mixed/speed/speed_result.pkl',
                 'out/tiny_mixed/speed/result.pkl',
                 'out/v2_mixed/speed/result.pkl'],
    names=['GateNet5-Mixed',
           'GateNet8-Mixed',
           'GateNet9-Mixed',
           'GateNet10-Mixed',
           'GateNet11-Mixed',
           'GateNet12-Mixed',
           'GateNet13-Mixed',
           'GateNet14-Mixed',
           'GateNet15-Mixed',
           'GateNet16-Mixed',
           'GateNet17-Mixed',
           'Tiny-Mixed',
           'V2-Mixed'
           ]
)

params_speed = params_speed_plot([1588617,
                                  619529,
                                  248265,
                                  723417,
                                  613337,
                                  285385,
                                  174025,
                                  285385,
                                  15867885,
                                  36341,
                                  244441,
                                  244441
                                  #                                  50676436,
                                  ],
                                 speed_files=['out/gatev5_mixed/speed/result.pkl',
                                              'out/gatev8_mixed/speed/result.pkl',
                                              'out/gatev9_mixed/speed/result.pkl',
                                              'out/gatev10_mixed/speed/result.pkl',
                                              'out/gatev11_mixed/speed/result.pkl',
                                              'out/gatev12_mixed/speed/result.pkl',
                                              'out/gatev13_mixed/speed/result.pkl',
                                              'out/gatev14_mixed/speed/speed_result.pkl',
                                              'out/gatev15_mixed/speed/result.pkl',
                                              'out/gatev16_mixed/speed/speed_result.pkl',
                                              'out/gatev17_mixed/speed/speed_result.pkl',
                                              'out/tiny_mixed/speed/result.pkl',
                                              #                                              'out/v2_mixed/speed/result.pkl'
                                              ],
                                 names=['GateNet5-Mixed',
                                        'GateNet8-Mixed',
                                        'GateNet9-Mixed',
                                        'GateNet10-Mixed',
                                        'GateNet11-Mixed',
                                        'GateNet12-Mixed',
                                        'GateNet13-Mixed',
                                        'GateNet14-Mixed',
                                        'GateNet15-Mixed',
                                        'GateNet16-Mixed',
                                        'GateNet17-Mixed',
                                        'Tiny-Mixed',
                                        #                                        'V2-Mixed'
                                        ]
                                 )

layers_speed = layers_speed_plot([6,
                                  9,
                                  6,
                                  8,
                                  10,
                                  10,
                                  7,
                                  10,
                                  4,
                                  8,
                                  7,
                                  10
                                  #                                  50676436,
                                  ],
                                 speed_files=['out/gatev5_mixed/speed/result.pkl',
                                              'out/gatev8_mixed/speed/result.pkl',
                                              'out/gatev9_mixed/speed/result.pkl',
                                              'out/gatev10_mixed/speed/result.pkl',
                                              'out/gatev11_mixed/speed/result.pkl',
                                              'out/gatev12_mixed/speed/result.pkl',
                                              'out/gatev13_mixed/speed/result.pkl',
                                              'out/gatev14_mixed/speed/speed_result.pkl',
                                              'out/gatev15_mixed/speed/result.pkl',
                                              'out/gatev16_mixed/speed/speed_result.pkl',
                                              'out/gatev17_mixed/speed/speed_result.pkl',
                                              'out/tiny_mixed/speed/result.pkl',
                                              #                                              'out/v2_mixed/speed/result.pkl'
                                              ],
                                 names=['GateNet5-Mixed',
                                        'GateNet8-Mixed',
                                        'GateNet9-Mixed',
                                        'GateNet10-Mixed',
                                        'GateNet11-Mixed',
                                        'GateNet12-Mixed',
                                        'GateNet13-Mixed',
                                        'GateNet14-Mixed',
                                        'GateNet15-Mixed',
                                        'GateNet16-Mixed',
                                        'GateNet17-Mixed',
                                        'Tiny-Mixed',
                                        #                                        'V2-Mixed'
                                        ]
                                 )
ps_plot.show(False)
params_speed.show(False)
layers_speed.show(False)
pr_basement_tuning.show(False)
pr_daylight_tuning.show(True)
