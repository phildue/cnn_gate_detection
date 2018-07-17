import os

from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()
models = ['baseline104x104-13x13+9layers',
          'baseline208x208-13x13+9layers',
          'baseline416x416-13x13+9layers',
          'baseline52x52-13x13+9layers',
          'bottleneck416x416-13x13+9layers',
          'bottleneck_narrow416x416-13x13+9layers',
          'bottleneck_narrow_strides416x416-13x13+9layers',
          'combined208x208-13x13+13layers',
          'grayscale416x416-13x13+9layers',
          'mobilenetV1416x416-13x13+9layers',
          'narrow416x416-13x13+9layers',
          'narrow_strides416x416-13x13+9layers',
          'narrow_strides_late_bottleneck416x416-13x13+9layers',
          'strides2416x416-13x13+9layers',
          'strides416x416-13x13+9layers']

models = [name for name in os.listdir('out/1807/')]
for model in models:
    preprocessing = None if 'gray' not in model else TransformGray()
    evalset(name='test',
            batch_size=24,
            # n_samples=48,
            model_src='out/1807/' + model,
            preprocessing=None,
            color_format='yuv',
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'])

for model in models:
    for iou_thresh in [0.4, 0.6, 0.8]:
        for min_box_area in [0.001, 0.025, 0.05, 0.1, 0.25]:
            evalmetric(name='test_iou{}-area{}'.format(iou_thresh, min_box_area),
                       min_box_area=min_box_area,
                       iou_thresh=iou_thresh,
                       batch_size=24,
                       model_src='out/1807/' + model,
                       color_format='yuv',
                       label_file='out/1807/' + model + '/test/test_results.pkl',
                       show=False)
