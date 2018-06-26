from modelzoo.evaluation import evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.gatenet.GateNet import GateNet
from run.evaluation.evalset import evalset
from utils.fileaccess.utils import load_file
import numpy as np

models = ['refnet52x52-3x3+4layers+16filters',
          'refnet52x52-3x3+4layers+32filters',
          'refnet52x52-3x3+4layers+64filters',
          'refnet52x52-3x3+6layers+16filters',
          'refnet52x52-3x3+6layers+32filters',
          'refnet52x52-3x3+6layers+64filters']
grids = [
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),

]
img_res = [
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
]
ious = [0.4, 0.6, 0.8]

for i, model in enumerate(models):
    evalset(name='',
            batch_size=8,
            model_src='out/2606/' + model,
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'],
            grid=grids[i],
            img_res=img_res[i],
            iou_thresh=0,
            n_samples=None)

for i, model in enumerate(models):
    conf_thresh = 0
    summary = load_file(model + '/summary.pkl')
    architecture = summary['architecture']
    grid = grids[i]
    model = GateNet.create_by_arch(norm=img_res, architecture=architecture,
                                   anchors=np.array([[[1, 1],
                                                      [1 / grid[0], 1 / grid[1]],  # img_h/img_w
                                                      [2 / grid[0], 2 / grid[1]],  # 0.5 img_h/ 0.5 img_w
                                                      [1 / grid[0], 3 / grid[0]],  # img_h / 0.33 img_w
                                                      [1 / grid[0], 2 / grid[0]]  # img_h / 0.5 img_w
                                                      ]]),
                                   batch_size=8,
                                   color_format='yuv',
                                   conf_thresh=conf_thresh,
                                   augmenter=None,
                                   weight_file=model + '/model.h5'
                                   )
    for iou in ious:
        evaluate_file(model, 'out/2606/' + model + '/results/result_0.pkl',
                      metrics=[MetricDetection(iou_thresh=iou, show_=False)],
                      verbose=True,
                      out_file_metric='out/2606/' + model + '/detect' + str(iou) + '.pkl')
