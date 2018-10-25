import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.evalcluster import eval
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()
result_files = [
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    # 'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_allview416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl'
]

titles = [
    'Random Placement',
    'Race Track',
    'Combined',
]

ObjectLabel.classes = ['gate']
ious = [0.4, 0.6, 0.8]
frame = pd.DataFrame()
frame['Name'] = titles
for iou in ious:
    tps = []
    fps = []
    fns = []
    boxes_true = []

    for i, f in enumerate(result_files):
        result_file = load_file(f)
        labels_pred = result_file['labels_pred']
        labels_true = result_file['labels_true']
        img_files = result_file['image_files']
        tp, fp, fn, b_true = eval(labels_true, labels_pred, iou_thresh=iou)

        tps.append(tp)
        fps.append([b for b in fp if b.confidence > 0.01])
        fns.append(fn)
        boxes_true.append(b_true)
    frame['TP' + str(iou)] = tps
    frame['FP' + str(iou)] = fps
    frame['FN' + str(iou)] = fns
    frame['True' + str(iou)] = boxes_true

plt.figure(figsize=(8, 3))
plt.subplot(1, 3, 1)
plt.xlabel('Confidence')
plt.ylabel('True Positives')
w = 1 / len(result_files)
c = []
for i, f in enumerate(result_files):
    c.append([tp.confidence for tp in frame['TP' + str(0.6)][i]])

plt.hist(c)

plt.subplot(1, 3, 2)
plt.xlabel('Confidence')
plt.ylabel('False Positives')
w = 1 / len(result_files)
c = []
for i, f in enumerate(result_files):
    c.append([fp.confidence for fp in frame['FP' + str(0.6)][i]])
plt.hist(c)

plt.subplot(1, 3, 3)
plt.xlabel('Confidence')
plt.ylabel('False Negatives')
w = .1 / len(result_files)
conf = np.zeros((len(result_files), 11))
for i, f in enumerate(result_files):
    for j, c in enumerate(np.linspace(0, 1, 11)):
        conf[i, j] = len(frame['True0.6'][i]) - len([tp for tp in frame['TP' + str(0.6)][i] if tp.confidence > c])
    plt.bar(np.linspace(0, 1, 11)-w+i*w, conf[i])

plt.legend(titles)
plt.show()
