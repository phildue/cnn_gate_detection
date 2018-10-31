import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.evalcluster import evalcluster_yaw_dist
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

result_files = [
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    # 'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_allview416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_arch2416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_arch_random416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_arch_race416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl'
]

titles = [
    'Random Placement',
    'Race Track',
    'Combined',
    'Ours',
    'yolov3_arch_random416x416',
    'yolov3_arch_race416x416',
]

ObjectLabel.classes = ['gate']
dist_bins = 10
angle_bins = 10
angles = np.linspace(0, 180, angle_bins)
distances = np.linspace(0, 12, dist_bins)
ious = [0.4, 0.6, 0.8]
frame = pd.DataFrame()
frame['Name'] = titles
for iou in ious:
    recalls = []
    fns = []
    tps = []
    for i, f in enumerate(result_files):
        result_file = load_file(f)
        labels_pred = result_file['labels_pred']
        labels_true = result_file['labels_true']
        img_files = result_file['image_files']
        fn, tp = evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh=0.9, n_bins_angle=angle_bins,
                                      n_bins_dist=dist_bins,
                                      iou_thresh=iou)

        recall = (tp / (fn + tp))
        recall[np.isnan(recall)] = 0.0
        recalls.append(recall)
        tps.append(tp)
        fns.append(fn)
    frame['Recall' + str(iou)] = recalls
    frame['TP' + str(iou)] = tps
    frame['FN' + str(iou)] = fns
print(frame.to_string())
recalls = frame['Recall0.6']
plt.figure(figsize=(9, 3))
plt.title('Recall per Cluster')
for i, r in enumerate(recalls):
    plt.subplot(1, len(titles), i + 1)
    plt.pcolor(r, cmap=plt.cm.viridis, vmin=0, vmax=1.0)
    if i < 3:
        plt.title(titles[i], fontsize=12)

    plt.xlabel('Relative Yaw Angle')
    plt.ylabel('Relative Distance')
    plt.yticks(np.arange(1, dist_bins, 2), np.round(distances[1::2]))
    plt.xticks(np.arange(2, angle_bins, 3), np.round(angles[2::3]))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)

plt.colorbar()
# plt.savefig('doc/thesis/fig/recall_yaw.png')

tp08 = frame['TP0.8']
tp06 = frame['TP0.6']
tp04 = frame['TP0.4']
tps = [tp04, tp06, tp08]

fn08 = frame['FN0.8']
fn06 = frame['FN0.6']
fn04 = frame['FN0.4']
fns = [fn04, fn06, fn08]
recall_front = []
recall_yaw = []
for i, tp in enumerate(tps):
    r_front_iou = []
    r_yaw_iou = []
    for j in range(len(tp)):
        fn_sum = (fns[i][j][:, 0] + fns[i][j][:, -1])
        tp_sum = (tp[j][:, 0] + tp[j][:, -1])
        r = tp_sum / (tp_sum + fn_sum)
        r[np.isnan(r)] = 0.0
        r_front_iou.append(r)
        fn_total = fn_sum
        tp_total = tp_sum
        fn_sum = np.sum(fns[i][j][:, 1:-1], 1)
        tp_sum = np.sum(tp[j][:, 1:-1], 1)
        r = tp_sum / (tp_sum + fn_sum)
        r[np.isnan(r)] = 0.0
        r_yaw_iou.append(r)
        fn_total += fn_sum
        tp_total += tp_sum
        fn_total = np.sum(fn_total)
        tp_total = np.sum(tp_total)
        print("{}: IoU={} Total {}/{} [{}]: r={}".format(titles[j],ious[i],tp_total,fn_total,tp_total + fn_total,tp_total/(tp_total+fn_total)))
    recall_yaw.append(r_yaw_iou)
    recall_front.append(r_front_iou)

plt.figure(figsize=(8, 3))

for j, r in enumerate(recall_front[1:], 1):
    plt.subplot(1, 2, j)
    plt.title('Recall at Yaw +- 20°C, IoU={}'.format(ious[j]))
    plt.xlim(-0.5, 10)
    plt.ylim(0, 1.0)
    w = 1 / len(result_files)
    # colors = [['#faa083', '#e04a1b', '#b33b16'], ['#ffdc8b', '#ffc12e', '#d18317'], ['#c4de8b', '#8abe17', '#466500']]
    for i in range(len(result_files)):
        plt.bar(distances - w + i * w, r[i], width=w, align='center')
    plt.xlabel('Relative Distance [m]')
    plt.ylabel('Recall')
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)
# plt.savefig('doc/thesis/fig/recall_front.png')

plt.figure(figsize=(8, 3))
for j, r in enumerate(recall_yaw[1:], 1):
    plt.subplot(1, 2, j)
    plt.title('Recall at Yaw  > 20 °C, IoU={}'.format(ious[j]))
    plt.xlim(-0.5, 10)
    plt.ylim(0, 1.0)
    w = 1 / len(ious)
    # colors = [['#faa083', '#e04a1b', '#b33b16'], ['#ffdc8b', '#ffc12e', '#d18317'], ['#c4de8b', '#8abe17', '#466500']]
    for i in range(len(result_files)):
        plt.bar(distances - w + i * w, r[i], width=w, align='center')
    plt.xlabel('Relative Distance [m]')
    plt.ylabel('Recall')
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)
# plt.savefig('doc/thesis/fig/recall_angle.png')

plt.show(True)
