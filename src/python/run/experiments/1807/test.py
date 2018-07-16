import os

from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()
#
for model in [name for name in os.listdir('out/1807/')]:
    preprocessing = None if 'gray' not in model else TransformGray()
    evalset(name='test',
            batch_size=24,
            model_src= 'out/1807/'+model,
            preprocessing=None,
            color_format='yuv',
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'])

for model in [name for name in os.listdir('out/1807/')]:
    for iou_thresh in [0.4, 0.6, 0.8]:
        for min_box_area in [0.05, 0.1, 0.25, 0.5, 0.7]:
            evalmetric(name='test',
                       min_box_area=min_box_area,
                       batch_size=24,
                       model_src='out/1807/'+model,
                       color_format='yuv',
                       label_file=model + '/test/test_iou{}-area{}_results.pkl'.format(iou_thresh,min_box_area),
                       show=False)
