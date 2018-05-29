import os
import sys

import numpy as np



PROJECT_ROOT = '/home/phil/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.utils import load
import matplotlib.pyplot as plt
from visualization.ap_position import group_by_pos, sort_results, mean_groups
from backend.plots.BaseStepPlot import BaseStepPlot
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall

def group_mean(group, bin_min, bin_max, n_bins=10):
    sorted = sort_results(group, bin_min, bin_max, n_bins)
    return mean_groups(sorted)


def plot_ap_bin(group, bins, xlabel, title='', output_path=None, fontsize=12,
                fig_size=(5, 4),
                block=True):
    p = BaseStepPlot(y_data=np.array(group), x_data=np.array(bins), x_label=xlabel, y_label='meanAP',
                     title=title,
                     size=fig_size, font_size=fontsize)
    p.show(block)
    p.save(output_path + 'meanAP-' + xlabel + '.png')

def sum_metrics(results):
    total = results
    for result in results:
        total = total + result
    return total

result_path = 'logs/yolo-gate-hard/'
N_BINS = 30
FONT_SIZE = 20
experiments = load(result_path + 'experiment_results_10000.pkl')

group_eucl, group_pitch, group_yaw, group_roll, _, _, _ = group_by_pos(experiments)

mean_eucl_04, bin_eucl_04 = group_mean(group_eucl, 5, 30, 60)
mean_pitch_04, bin_pitch_04 = group_mean(group_pitch, -np.pi / 2, np.pi / 2, N_BINS)
mean_yaw_04, bin_yaw_04 = group_mean(group_yaw, -np.pi / 2, np.pi / 2, N_BINS)
mean_roll_04, bin_roll_04 = group_mean(group_roll, -np.pi / 2, np.pi / 2, N_BINS)
metrics_04 = [e[0] for e in experiments]
total = sum_metrics(metrics_04)
values = [total[k] for k in reversed(sorted(total.keys()))]

interp_04, ap_04 = EvaluatorPrecisionRecall.interp(values)

experiments = load(result_path + 'experiment_results_iou06.pkl')
group_eucl, group_pitch, group_yaw, group_roll, _, _, _ = group_by_pos(experiments)

mean_eucl_06, bin_eucl_06 = group_mean(group_eucl, 5, 30, 60)
# mean_pitch_06, bin_pitch_06 = group_mean(group_pitch, 5, 30, N_BINS)
# mean_yaw_06, bin_yaw_06 = group_mean(group_yaw, 5, 30, N_BINS)
# mean_roll_06, bin_roll_06 = group_mean(group_roll, 5, 30, N_BINS)

metrics_06 = [e[0] for e in experiments]
total = sum_metrics(metrics_06)
values = [total[k] for k in reversed(sorted(total.keys()))]
interp_06, ap_06 = EvaluatorPrecisionRecall.interp(values)

mean_eucl = np.array([[mean_eucl_04], mean_eucl_06])
bin_eucl = np.array([[bin_eucl_04], bin_eucl_06])
# mean_pitch = np.array([[mean_pitch_04], mean_pitch_06])
# mean_yaw = np.array([[mean_yaw_04], mean_yaw_06])
# mean_roll = np.array([[mean_roll_04], mean_roll_06])

plt.figure(figsize=(24, 6))
h04, = plt.step(bin_eucl_04, mean_eucl_04, label='mid')
h06, = plt.step(bin_eucl_06, mean_eucl_06, label='mid')

plt.legend([h04, h06], ['Iou=0.4', 'Iou=0.6'], fontsize=FONT_SIZE)
plt.title('N=10 000', fontsize=FONT_SIZE)
plt.xlabel('Euclidian Distance [m]', fontsize=FONT_SIZE)
plt.ylabel('Average Precision', fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.grid(True, axis='both', which='both')
plt.savefig('../../doc/poster2/fig/meanAP-eucl.png')
plt.show(block=False)

plt.figure(figsize=(24, 6))
plt.subplot(1, 2, 2)
# h_yaw, = plt.plot(bin_yaw_04, mean_yaw_04, 'o--')
# h_roll, = plt.plot(bin_roll_04, mean_roll_04,  'o:')
# h_pitch, = plt.plot(bin_pitch_04, mean_pitch_04,  'x--')
h_yaw, = plt.step(bin_yaw_04, mean_yaw_04)
h_roll, = plt.step(bin_roll_04, mean_roll_04, linestyle=':')
h_pitch, = plt.step(bin_pitch_04, mean_pitch_04, linestyle='--')

plt.legend([h_yaw, h_roll, h_pitch], ['Yaw-Angle', 'Roll-Angle', 'Pitch-Angle'], fontsize=FONT_SIZE)
plt.title('N=10 000, IoU=0.4', fontsize=FONT_SIZE)
plt.xlabel('Angle [°]', fontsize=FONT_SIZE)
plt.ylabel('Average Precision', fontsize=FONT_SIZE)
# plt.ylim(0.0, 0.7)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.grid(True, axis='both', which='both')
# plt.savefig('../../doc/poster2/fig/meanAP-angle.png')

# plt.show(block=False)
plt.subplot(1, 2, 1)

# plt.figure(figsize=(11, 5))
h04, = plt.plot(interp_04[1], interp_04[0], '--')

h06, = plt.plot(interp_06[1], interp_06[0], '--')
plt.annotate('meanAP_40= ' + str(np.round(ap_04, 2)), xy=(0.6, 0.5),
             xytext=(0.6, 0.5), fontsize=FONT_SIZE)

plt.annotate('\nmeanAP_60= ' + str(np.round(ap_06, 2)), xy=(0, 0),
             xytext=(0, 0), fontsize=FONT_SIZE)

plt.legend([h04, h06], ['Iou=0.4', 'Iou=0.6'], fontsize=FONT_SIZE)
plt.title('N=10 000', fontsize=FONT_SIZE)
plt.xlabel('Recall', fontsize=FONT_SIZE)
plt.ylabel('Precision', fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.grid(True, axis='both', which='both')
plt.savefig('../../doc/poster2/fig/angle-pr.png')
plt.show()
