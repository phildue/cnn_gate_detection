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


def avg_pr_per_image_file(src_file):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_pr, mean_rec = average_precision_recall(detection_result)
    return mean_pr, mean_rec


def avg_pr_file(src_file):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']

    detection_result = [ResultByConfidence(d) for d in detection_result]
    detection_sum = detection_result[0]
    for d in detection_result[1:]:
        detection_sum += d
    precision = np.zeros((10,))
    recall = np.zeros((10,))
    for j, c in enumerate(np.round(np.linspace(0.0, 0.9, 10), 2)):
        precision[j] = detection_sum.results[c].precision
        recall[j] = detection_sum.results[c].recall

    return precision, recall


def pr_plot(files, legend, title, line_style=None, y_range=(0.5, 1.0)):
    recalls = []
    precisions = []
    for f in files:
        mean_p, mean_r = avg_pr_per_image_file(f)
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


def performance_speed_plot(performance_files, speed_files, names, linestyles=None):
    times = []
    performances = []
    for i in range(len(performance_files)):
        speed_file = speed_files[i]
        performance_file = performance_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        times.append([t_pred])

        performance_file_cont = load_file(performance_file)
        detection_result = performance_file_cont['results']['MetricDetection']
        detection_result = [ResultByConfidence(d) for d in detection_result]
        precision, _ = avg_pr_per_image_file(performance_file)
        #        print("MAP:{}:{}+-{}".format(names[i], np.mean(precision), np.std(precision)))
        #        print("Speed: {}:{}+-{}".format(names[i], t_pred, np.std(speed_file_cont['results_pred'][1:])))
        performances.append([np.mean(precision)])

    return BaseMultiPlot(x_data=times, x_label='Inference Time [s]',
                         y_data=performances, y_label='Mean Average Precision',
                         line_style=linestyles,
                         legend=names,
                         dark=True)


def params_speed_plot(params, speed_files, names, linestyle=None):
    times = []
    performances = []
    for i in range(len(speed_files)):
        speed_file = speed_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        times.append([t_pred])

        performances.append([params[i]])

    return BaseMultiPlot(y_data=times, y_label='Inference Time [s]',
                         x_data=performances, x_label='Weights',
                         line_style=linestyle,
                         legend=names,
                         dark=True,
                         size=(8, 4.5))


def layers_speed_plot(n_layers, speed_files, names, linestyle=None):
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
                         line_style=linestyle,
                         legend=names,
                         dark=True)


def performance_weights(performance_files, params, names, linestyle=None):
    performances = []
    params_l = []
    # symbols = ['x', 'o', '*']
    # linestyle = []
    for i in range(len(performance_files)):
        params_l.append([params[i]])
        performance_file = performance_files[i]
        performance_file_cont = load_file(performance_file)
        detection_result = performance_file_cont['results']['MetricDetection']
        detection_result = [ResultByConfidence(d) for d in detection_result]
        precision, _ = avg_pr_per_image_file(performance_file)
        # print("{}:{}+-{}".format(names[i], np.mean(precision), np.std(precision)))
        performances.append([np.mean(precision)])
        # linestyle.append(symbols[i % len(symbols)])

    return BaseMultiPlot(x_data=params_l, x_label='Weights',
                         y_data=performances, y_label='Mean Average Precision',
                         line_style=linestyle,
                         legend=names,
                         dark=True,
                         size=(8, 4.5)
                         )


cd_work()

daylight_result_files = [

    'out/gatev8_mixed/results/daylight--020.pkl',
    'out/gatev9_mixed/results/daylight--020.pkl',
    'out/gatev10_mixed/results/daylight--020.pkl',
    'out/gatev11_mixed/results/daylight--018.pkl',
    'out/gatev12_mixed/results/daylight--018.pkl',
    'out/gatev13_mixed/results/daylight--018.pkl',
    'out/gatev14_mixed/results/daylight--018.pkl',
    'out/gatev15_mixed/results/test_2--009.pkl',
    'out/gatev16_mixed/results/test_2--019.pkl',
    'out/gatev17_mixed/results/set_2--017.pkl',
    'out/gatev18_mixed/results/set_2--017.pkl',
    'out/gatev19/results/set_2--017.pkl',
    'out/gatev20/results/set_2--017.pkl',
    # 'out/tiny_mixed/results/daylight--023.pkl',
    'out/v2_mixed/results/daylight--019.pkl'
]

speed_results = [

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
    'out/gatev18_mixed/speed/speed_result.pkl',
    'out/gatev19/speed/speed_result.pkl',
    'out/gatev20/speed/speed_result.pkl',
    # 'out/tiny_mixed/speed/result.pkl',
    'out/v2_mixed/speed/result.pkl']

legend = None

symbols = [

    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    'x-',
    # 'o-',
    'o-', ]

params_speed = params_speed_plot([619529,
                                  248265,
                                  723417,
                                  613337,
                                  285385,
                                  174025,
                                  285385,
                                  36341,
                                  244441,
                                  687577,
                                  723929,
                                  542921,
                                  640073,
                                  # 15867885
                                  50676436,
                                  ],
                                 speed_files=speed_results,
                                 names=legend,
                                 linestyle=symbols
                                 )

params_speed2 = params_speed_plot([619529,
                                   248265,
                                   723417,
                                   613337,
                                   285385,
                                   174025,
                                   285385,
                                   36341,
                                   244441,
                                   687577,
                                   723929,
                                   542921,
                                   640073,
                                   # 15867885
                                   # 50676436,
                                   ],
                                  speed_files=speed_results[:-1],
                                  names=legend,
                                  linestyle=symbols
                                  )
params_perf = performance_weights(performance_files=daylight_result_files[:-1],
                                  params=[619529,
                                          248265,
                                          723417,
                                          613337,
                                          285385,
                                          174025,
                                          285385,
                                          36341,
                                          244441,
                                          687577,
                                          723929,
                                          542921,
                                          640073,
                                          # 15867885,
                                          # 50676436,
                                          ],
                                  names=None,
                                  linestyle=symbols)

params_perf2 = performance_weights(performance_files=daylight_result_files,
                                   params=[619529,
                                           248265,
                                           723417,
                                           613337,
                                           285385,
                                           174025,
                                           285385,
                                           36341,
                                           244441,
                                           687577,
                                           723929,
                                           542921,
                                           640073,
                                           # 15867885,
                                           50676436,
                                           ],
                                   names=legend,
                                   linestyle=symbols)

# ps_plot = performance_speed_plot(performance_files=daylight_result_files, speed_files=speed_results, names=legend,
#                                  linestyles=symbols)

params_speed.save('doc/presentation/mid-term/fig/params_speed.png', transparent=True)
params_speed2.save('doc/presentation/mid-term/fig/params_speed_close.png', transparent=True)
params_perf.save('doc/presentation/mid-term/fig/params_perf.png', transparent=True)
params_perf2.save('doc/presentation/mid-term/fig/params_perf_close.png', transparent=True)
# ps_plot.save('doc/presentation/mid-term/fig/perf_speed.png',transparent=True)

params_perf.show(False)
params_perf2.show(False)
params_speed.show(False)
params_speed2.show(True)
# ps_plot.show(True)
