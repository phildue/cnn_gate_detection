import matplotlib.pyplot as plt
import numpy as np

from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import sum_results, average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
results_root = 'out/thesis/snake/'
image_root = 'resource/ext/samples/'
datasets = [
    # 'real_test_labeled',
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
    # 'iros2018_course_final_simple_17gates'
]

titles = [
    # 'Combined',
    'Cyberzoo',
    'Basement',
    'Hallway',
    # 'Simulation'
]

box_sizes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0]
img_area = 640 * 480
ious = [0.6]
plots = []
for i, d in enumerate(datasets):
    curves = []
    for iou in ious:
        results_sums = []
        for n in range(5):
            box_min = box_sizes[0]
            box_max = box_sizes[-1]
            result_file = 'out/thesis/snake/{}_boxes{}-{}_iou{}_i{}.pkl'.format(d, box_min, box_max, iou, n)

            results = load_file(result_file)
            resultsByConf = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            results_sum = sum_results(resultsByConf)
            results_sums.append(results_sum)

        average_pr = average_precision_recall(results_sums, np.linspace(0, 1.0, 11))
        # mean = mean_results(results_sums)
        # average = mean.precisions, mean.recalls
        curves.append(average_pr)
    plots.append(curves)

plt.figure(figsize=(8, 6))
plt.suptitle("Average Precision Recall at an IoU of 0.6", fontsize=12)
for i, p in enumerate(plots, 1):
    # plt.subplot(2, 2, i)
    for j in range(len(ious)):
        mean_p, mean_r, std_p, std_r = p[j]
        # mean_p, mean_r = p[j]
        # plt.plot(xdata=mean_p, ydata=mean_r, linestyle='o--')
        plt.errorbar(y=mean_p, x=mean_r, yerr=std_p, linestyle='-.',
                     uplims=True, lolims=True)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.grid(axis='both', which='minor')
# plt.title(titles[i - 1], fontsize=12)
plt.legend(titles)
plt.ylim(-0.01, 1.01)
plt.xlim(-0.01, 1.01)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.show(False)
plt.savefig('doc/thesis/fig/snake_results_real.png')

plots = []
for i, d in enumerate(datasets):
    curves = []
    for iou in ious:
        average_precisions = []
        std_precisions = []
        for k in range(len(box_sizes) - 1):
            results_sums = []
            for n in range(5):
                box_min = box_sizes[k]
                box_max = box_sizes[k + 1]
                result_file = 'out/thesis/snake/{}_boxes{}-{}_iou{}_i{}.pkl'.format(d, box_min, box_max, iou, n)

                results = load_file(result_file)
                resultsByConf = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
                results_sum = sum_results(resultsByConf)
                results_sums.append(results_sum)

            mean_p, _, std_p, _ = average_precision_recall(results_sums)
            # mean_p = mean_results(results_sums).true_positives
            average_precisions.append(np.mean(mean_p))
            std_precisions.append(np.mean(std_p))
        curves.append((np.array(average_precisions), np.array(std_precisions)))
    plots.append(curves)

plt.figure(figsize=(8, 6))
plt.suptitle("Performance with respect to Object Size", fontsize=12)
for i, p in enumerate(plots, 1):
    plt.subplot(2, 2, i)
    for j in range(len(ious)):
        mean_p, std_p = p[j]
        # mean_p, mean_r = p[j]
        # plt.plot(xdata=mean_p, ydata=mean_r, linestyle='o--')
        plt.bar(np.array(box_sizes[:-1]) - 0.03 + j * 0.03, mean_p, width=0.03)
    plt.xlabel("Object Size as Fraction of Image Size", fontsize=12)
    plt.ylabel("Mean Average Precision", fontsize=12)
    plt.title(titles[i - 1], fontsize=12)
    # plt.legend(['IoU = {}'.format(k) for k in [0.4, 0.6, 0.8]])
    # plt.ylim(-0.01, 1.01)
    plt.xlim(-0.04, 1.4)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/snake_results_real_size.png')
plt.show(True)
