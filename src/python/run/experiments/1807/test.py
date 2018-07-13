import os

from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()
#
for model in [name for name in os.listdir('out/1807/') if os.path.isdir(name)]:
    preprocessing = None if 'gray' not in model else TransformGray()
    evalset(name='test',
            batch_size=24,
            model_src=model,
            preprocessing=None,
            color_format='yuv',
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'])

for model in [name for name in os.listdir('out/1807/') if os.path.isdir(name)]:
    evalmetric(name='test',
               batch_size=24,
               model_src=model,
               color_format='yuv',
               label_file=model + '/test/test_results.pkl',
               show=False)
