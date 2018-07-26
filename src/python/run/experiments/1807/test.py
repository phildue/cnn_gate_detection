import os

from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()
models = [
    'baseline104x104-13x13+9layers',
    'baseline208x208-13x13+9layers',
    'baseline416x416-13x13+9layers',
    'baseline52x52-13x13+9layers',
    # 'bottleneck416x416-13x13+9layers',
    # 'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    # 'grayscale416x416-13x13+9layers',
    # 'mobilenetV1416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    # 'narrow_strides_late_bottleneck416x416-13x13+9layers',
    # 'strides2416x416-13x13+9layers',
    # 'strides416x416-13x13+9layers'
]

# models = [name for name in os.listdir('out/1807/')]
# for model in models:
#     preprocessing = None if 'gray' not in model else TransformGray()
#     evalset(name='boxrange_flight',
#             batch_size=24,
#             # n_samples=48,
#             model_src='out/1807/' + model,
#             preprocessing=None,
#             color_format='yuv',
#             image_source=['resource/ext/samples/daylight_flight/'])

box_range = [0.25, 0.5, 0.5, 1.0]
for model in models:
    for iou_thresh in [0.4, 0.6]:
        for i in range(len(box_range) - 1):
            evalmetric(name='range_iou{}-area{}_result_metric'.format(iou_thresh, box_range[i]),
                       min_box_area=box_range[i],
                       max_box_area=box_range[i + 1],
                       iou_thresh=iou_thresh,
                       batch_size=24,
                       model_src='out/1807/' + model,
                       color_format='yuv',
                       label_file='out/1807/' + model + '/test/test_results.pkl',
                       show=False)
