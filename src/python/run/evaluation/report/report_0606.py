import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
from matplotlib import pyplot as plt


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
    symbols = ['.', 'x', 'o', '*']
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
                         line_style=['x', 'o', '*', 'v', '<'],
                         legend=names,
                         y_lim=(0, 1.0),
                         size=(10, 5))


def speed_width(width, speed):
    return BasePlot(x_data=width, x_label='N_Filters',
                    y_data=speed, y_label='Inference Time [ms]',
                    line_style='x', title='1Layer at 128x128')


def speed_layers(layers, speed):
    return BasePlot(x_data=layers, x_label='N_Layers',
                    y_data=speed, y_label='Inference Time [ms]',
                    line_style='x', title='16Kernels at 128x128')


def speed_resolution(resolution, speed):
    plt.figure()
    plt.xticks([1, 2, 3, 4, 5, 6], ['16x16', '32x32', '64x64', '128x128', '256x256', '512x512'])
    plt.plot(resolution, speed, 'x')
    plt.title('1Layer 32Kernels')
    plt.ylabel('Inference Time [ms]')
    plt.xlabel('Resolution')
    plt.show(False)


cd_work()

performance_lay = performance_layers(
    [['out/v2_mixed/results/daylight--019.pkl'
      ],
     ['out/gatev10_mixed/results/daylight--020.pkl'
      ],
     [
         'out/gatev37/results/set_1-040.pkl',
         'out/gatev40/results/set_1-040.pkl',
         'out/gatev41/results/set_1-040.pkl',
         'out/gatev46/results/set_1-040.pkl',

     ],
     [
         'out/gatev39/results/set_1-040.pkl',
         'out/gatev42/results/set_1-040.pkl',
         'out/gatev43/results/set_1-040.pkl',
         'out/gatev44/results/set_1-040.pkl',
         'out/gatev47/results/set_1-040.pkl',
     ],
     [
         'out/gatev45/results/set_1-040.pkl',
         'out/gatev48/results/set_1-040.pkl',
         'out/gatev49/results/set_1-040.pkl',
         'out/gatev50/results/set_1-040.pkl',
         'out/gatev51/results/set_1-040.pkl',
     ]
     ],
    np.array([[23],
              [9],
              [
                  9,
                  7,
                  5,
                  3,
              ],
              [
                  9,
                  7,
                  5,
                  4,
                  2

              ],
              [
                  9,
                  7,
                  5,
                  3,
                  2
              ]
              ]),
    ['YoloV2-416x416',
     'GateNet-416x416',
     'GateNet-104x104',
     'GateNet-52x52',
     'GateNet-26x26'
     ]
)

speed_width([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128
], [27.49,
    28.45,
    29,
    28.36,
    30.3,
    31.18,
    32.55,
    30.88,
    33.41,
    34.14,
    35.4,
    33.85,
    36.5,
    37.24,
    38.72,
    37.66,
    48.44,
    71.58,
    118.08,
    ]
).show(False)

speed_layers([1, 2, 4, 5, 6, 7, 8, 10],
             [36.6, 126.52, 306.48, 396.45, 486.6, 580.43, 677.7, 856.36, ]).show(False)

speed_resolution([1, 2, 3, 4, 5, 6],
                 [1.91,
                  7.06,
                  26.95,
                  99.75,
                  390.62,
                  1559.67
                  ])

performance_lay.show(True)
