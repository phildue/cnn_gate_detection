import os

from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()
models = [
    'baseline104x104-13x13+9layers',
    'baseline416x416-13x13+9layers',
    'bottleneck416x416-13x13+9layers',
    'bottleneck_narrow416x416-13x13+9layers',
    'narrow416x416-13x13+9layers',
    'narrow_strides416x416-13x13+9layers',
    'narrow_strides_late_bottleneck416x416-13x13+9layers',
    'strides416x416-13x13+9layers'
]

box_range = [0.01, 0.05, 0.1, 0.15, 0.25, 1.0]
iou_threshs = 0.4, 0.6
for model in models:
    for iou_thresh in iou_threshs:
        for i in range(len(box_range) - 1):
            evalmetric(name='iou{}-area{}'.format(iou_thresh, box_range[i]),
                       min_box_area=box_range[i],
                       max_box_area=box_range[i + 1],
                       iou_thresh=iou_thresh,
                       batch_size=24,
                       model_src='out/1807/' + model,
                       color_format='yuv',
                       result_path='out/2507/areatest/' + model + '/',
                       label_file='out/1807/' + model + '/test/test_results.pkl',
                       show=False)
