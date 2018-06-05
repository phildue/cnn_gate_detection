import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work


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


def performance_resolution(performance_files, resolution, names):
    performances = []
    symbols = ['x', 'o', '*']
    linestyle = []
    resolutions = []
    for i in range(len(performance_files)):
        resolutions.append([resolution[i][0] * resolution[i][1]])
        performance_file = performance_files[i]

        precision, _ = avg_pr_per_image_file(performance_file)
        performances.append([np.mean(precision)])
        linestyle.append(symbols[i % len(symbols)])

    return BaseMultiPlot(x_data=resolutions, x_label='N_Pixel',
                         y_data=performances, y_label='Mean Average Precision',
                         line_style=linestyle,
                         legend=names)


def performance_layers(performance_files, layers, names):
    p = []
    for f in performance_files:
        performances = []
        for i in range(len(f)):
            performance_file = f[i]

            precision, _ = avg_pr_per_image_file(performance_file)
            performances.append(np.mean(precision))
        p.append(performances)

    return BaseMultiPlot(x_data=layers, x_label='N_Layers',
                         y_data=p, y_label='Mean Average Precision',
                         line_style=['x', 'o', '*'],
                         legend=names,
                         y_lim=(0.5, 1.0))


def speed_layers(n_layers, speed_files, names, linestyle=None):
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
                         legend=names)


def speed_resolution(resolution, speed_files, names, linestyle=None):
    times = []
    resolutions = []
    for i in range(len(speed_files)):
        speed_file = speed_files[i]
        speed_file_cont = load_file(speed_file)
        t_pred = np.mean(speed_file_cont['results_pred'][1:])
        print(np.std(speed_file_cont['results_pred'][1:]))
        times.append([t_pred])

        resolutions.append([resolution[i][0] * resolution[i][1]])

    return BaseMultiPlot(x_data=times, x_label='Inference Time [s]',
                         y_data=resolutions, y_label='Layers',
                         line_style=linestyle,
                         legend=names)


def speed_width():
    pass


cd_work()

performance_res = performance_resolution([
    'out/gatev10_mixed/results/daylight--020.pkl',
    'out/gatev37/results/set_1-020.pkl',
    'out/gatev39/results/set_1-020.pkl',
    'out/gatev45/results/set_1-020.pkl',
],
    [
        (416, 416),
        (104, 104),
        (52, 52),
        (26, 26),

    ],
    ['416x416',
     '104x104',
     '52x52',
     '26x26']
)

performance_lay = performance_layers([[
    'out/gatev37/results/set_1-020.pkl',
    'out/gatev40/results/set_1-020.pkl',
    'out/gatev41/results/set_1-020.pkl',
],
    [
        'out/gatev39/results/set_1-020.pkl',
        'out/gatev42/results/set_1-020.pkl',
        'out/gatev43/results/set_1-020.pkl',
        'out/gatev44/results/set_1-020.pkl',
    ],
    [
        'out/gatev45/results/set_1-020.pkl',
    ]
],
    np.array([[
        9,
        7,
        5,

    ],
        [
            9,
            7,
            5,
            3

        ],
        [
            9,
        ]
    ]),
    ['104x104',
     '52x52',
     '26x26'
     ]
)

performance_res.show(False)
performance_lay.show(True)
