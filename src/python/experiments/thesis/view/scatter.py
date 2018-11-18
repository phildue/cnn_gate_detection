import matplotlib.pyplot as plt
import pandas as pd

from evaluation.evalcluster import evalscatter_wh
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
result_files = [
    ['out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_jevois_cyberzoo/predictions.pkl',
     'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_jevois_basement/predictions.pkl',
     'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_jevois_hallway/predictions.pkl'
     ],
    ['out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_jevois_cyberzoo/predictions.pkl',
     'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_jevois_basement/predictions.pkl',
     'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_jevois_hallway/predictions.pkl'
     ],
    ['out/thesis/datagen/yolov3_allview416x416_i00/test_jevois_cyberzoo/predictions.pkl',
     'out/thesis/datagen/yolov3_allview416x416_i00/test_jevois_basement/predictions.pkl',
     'out/thesis/datagen/yolov3_allview416x416_i00/test_jevois_hallway/predictions.pkl'
     ]
]

titles = [
    'Random Placement',
    'Race Track',
    'Combined',
]

frame = pd.DataFrame()
frame['Name'] = titles
ious = [0.4, 0.6, 0.8]

for iou in ious:
    tps = []
    fps = []
    fns = []
    for i, d in enumerate(result_files, 0):
        tp_model = []
        fp_model = []
        fn_model = []
        for p in d:
            result_file = load_file(p)
            labels_pred = result_file['labels_pred']
            labels_true = result_file['labels_true']
            img_files = result_file['image_files']
            tp, fp, fn = evalscatter_wh(labels_true, labels_pred, conf_thresh=0.5, iou_thresh=iou)
            tp_model.extend(tp)
            fp_model.extend(fp)
            fn_model.extend(fn)
        tps.append(tp_model)
        fps.append(fp_model)
        fns.append(fn_model)
    frame['True Positives{}'.format(iou)] = tps
    frame['False Positives{}'.format(iou)] = fps
    frame['False Negatives{}'.format(iou)] = fns

plt.figure(figsize=(8, 3))
for i, m in enumerate(result_files):
    plt.subplot(1, 3, i + 1)
    plt.title(titles[i], fontsize=12)
    fns = frame['False Negatives0.4'][i]

    x = [fn[0] for fn in fns]
    y = [fn[1] for fn in fns]
    plt.plot(x, y, 'xc')


    tps = frame['True Positives0.4'][i]
    x = [tp[0] for tp in tps]
    y = [tp[1] for tp in tps]
    plt.plot(x, y, 'ob')

    tps = frame['True Positives0.8'][i]
    x = [tp[0] for tp in tps]
    y = [tp[1] for tp in tps]
    plt.plot(x, y, 'sr')

    plt.xlabel('Width', fontsize=12)
    plt.ylabel('Height', fontsize=12)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.legend(['FN', 'TP0.4', 'TP0.8'])
plt.savefig('doc/thesis/fig/view_scatter.png')
plt.show(True)
