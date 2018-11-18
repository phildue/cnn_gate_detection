import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import sum_results, average_precision_recall
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
    'SnakeGate-Cyberzoo',
    'SnakeGate-Basement',
    'SnakeGate-Hallway',
    'Network-Cyberzoo',
    'Network-Basement',
    'Network-Hallway',
]

ious = [0.4, 0.6, 0.8]
frame = pd.DataFrame()
frame['Names'] = titles
for iou in ious:
    ap = []
    mean_recs = []
    for i, d in enumerate(datasets):
        results_sums = []
        for n in range(5):
            result_file = 'out/thesis/snake/test_' + d + '_results_iou' + str(iou) + '_' + str(n) + '.pkl'.format(d,
                                                                                                                  iou,
                                                                                                                  n)

            results = load_file(result_file)
            results_sums.append(sum_results(results['results']))

        mean_pr, mean_rec, std_pr, std_rec = average_precision_recall(results_sums, np.linspace(0, 1.0, 11))

        ap.append(mean_pr)
        mean_recs.append(mean_rec)

    for i, d in enumerate(datasets):
        results_sums = []
        for n in range(0, 5):
            model_dir = 'out/thesis/datagen/yolov3_blur416x416' + '_i0{}'.format(n)
            result_file = model_dir + '/test_{}/results_iou{}.pkl'.format(d, iou)
            try:
                results = load_file(result_file)
                results_sums.append(sum_results(results['results']))
            except FileNotFoundError:
                print("Not found: {}".format(result_file))

        mean_pr, mean_rec, std_pr, std_rec = average_precision_recall(results_sums, np.linspace(0, 1.0, 11))

        ap.append(mean_pr)
        mean_recs.append(mean_rec)

    frame['Precision{}'.format(iou)] = ap
    frame['Recall{}'.format(iou)] = mean_recs

plt.figure(figsize=(8, 6))
plt.suptitle("Average Precision Recall at an IoU of 0.6", fontsize=12)
for i, p in enumerate(titles):
    plt.plot(frame['Recall0.6'][i], frame['Precision0.6'][i], 'x--')
    # plt.errorbar(y=mean_p, x=mean_r, yerr=std_p, linestyle='-.',
    #              uplims=True, lolims=True)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.grid(axis='both', which='minor')
# plt.title(titles[i - 1], fontsize=12)
plt.legend(titles)
plt.ylim(-0.01, 1.01)
plt.xlim(-0.01, 1.01)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)

plt.figure(figsize=(8, 6))
plt.suptitle("Comparison against Baseline", fontsize=12)
w = 1 / len(titles)
bars1 = []
bars2 = []
for iou in ious:
    bar1 = np.mean([np.mean(frame['Precision{}'.format(iou)][0]), np.mean(frame['Precision{}'.format(iou)][1]),
                    np.mean(frame['Precision{}'.format(iou)][2])], 0)
    bars1.append(bar1)
    bar2 = np.mean([np.mean(frame['Precision{}'.format(iou)][3]), np.mean(frame['Precision{}'.format(iou)][4]),
                    np.mean(frame['Precision{}'.format(iou)][5])], 0)

    bars2.append(bar2)

plt.bar(np.arange(len(ious)), bars1, width=w)
plt.bar(np.arange(len(ious)) + w, bars2, width=w)

plt.xticks(np.arange(len(ious)), ious)
plt.xlabel('Intersection Over Union')
plt.ylabel('Average Precision')
plt.ylim(0, 0.9)
# plt.xlabel("Recall", fontsize=12)
# plt.ylabel("Precision", fontsize=12)
# plt.grid(axis='both', which='minor')
# plt.title(titles[i - 1], fontsize=12)
plt.legend(['SnakeGate', 'Network'])
# plt.ylim(-0.01, 1.01)
# plt.xlim(-0.01, 1.01)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/comp_baseline.png')
plt.show(True)
