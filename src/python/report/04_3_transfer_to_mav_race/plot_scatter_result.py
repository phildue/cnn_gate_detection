import matplotlib.pyplot as plt
import pandas as pd

from evaluation.evaluation import evalscatter_wh
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'ewfo_sim',
    'randomview',
    'racecourt',
    'racecourt_allviews',
    'randomview_and_racecourt_allviews'
]

dataset = 'iros2018_course_final_simple_17gates'

titles = [
    'Frontal View',
    'Random Placement',
    'Simulated Flight',
    'Simulated Flight All Views',
    'Combined',
]

frame = pd.DataFrame()
frame['Name'] = titles
ious = [0.4, 0.6, 0.8]
n_iterations = 1
for iou in ious:
    tps = []
    fps = []
    fns = []
    for i, m in enumerate(models, 0):
        for it in range(n_iterations):
            result_file = load_file('out/{}_i0{}/test_{}/predictions.pkl'.format(m,it, dataset))
            labels_pred = result_file['labels_pred']
            labels_true = result_file['labels_true']
            img_files = result_file['image_files']
            tp, fp, fn = evalscatter_wh(labels_true, labels_pred, conf_thresh=0.3, iou_thresh=iou)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
    frame['True Positives{}'.format(iou)] = tps
    frame['False Positives{}'.format(iou)] = fps
    frame['False Negatives{}'.format(iou)] = fns

plt.figure(figsize=(8, 3))
for i, m in enumerate(models):
    plt.subplot(1, len(models), i + 1)
    plt.title(titles[i], fontsize=12)
    fns = frame['False Negatives0.6'][i]

    x = [fn[0] for fn in fns]
    y = [fn[1] for fn in fns]
    plt.plot(x, y, 'xc')

    tps = frame['False Positives0.6'][i]
    x = [tp[0] for tp in tps]
    y = [tp[1] for tp in tps]
    plt.plot(x, y, '.r')

    tps = frame['True Positives0.6'][i]
    x = [tp[0] for tp in tps]
    y = [tp[1] for tp in tps]
    plt.plot(x, y, '.g')

    plt.xlabel('Width', fontsize=12)
    plt.ylabel('Height', fontsize=12)
    plt.ylim(0,416)
    plt.xlim(0,416)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.legend(['FN', 'FP0.6', 'TP0.6'])
# plt.savefig('doc/thesis/fig/view_scatter.png')
plt.show(True)
