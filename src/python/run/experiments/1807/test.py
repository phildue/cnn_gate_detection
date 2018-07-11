from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.workdir import cd_work

cd_work()

for model in [
    # 'out/1807/gatenet-strided416x416-13x13+9layers+pyramid',
    'out/1807/gatenet416x416-13x13+9layers+pyramid',
    # 'out/1807/graygatenet416x416-13x13+9layers+pyramid',
    'out/1807/mobilegatenet416x416-13x13+9layers+pyramid',
    'out/1807/wr_basic_gatenet416x416-13x13+10layers+pyramid',
    # 'out/1807/wr_inception_gatenet416x416-13x13+10layers+pyramid'
]:
    # preprocessing = None if 'gray' not in model else TransformGray()
    evalset(name='test',
            batch_size=24,
            model_src=model,
            preprocessing=None,
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'])

for model in [
    'out/1807/gatenet-strided416x416-13x13+9layers+pyramid',
    'out/1807/gatenet416x416-13x13+9layers+pyramid',
    # 'out/1807/graygatenet416x416-13x13+9layers+pyramid',
    'out/1807/mobilegatenet416x416-13x13+9layers+pyramid',
    'out/1807/wr_basic_gatenet416x416-13x13+10layers+pyramid',
    # 'out/1807/wr_inception_gatenet416x416-13x13+10layers+pyramid'
]:
    evalmetric(name='test',
               batch_size=24,
               model_src=model,
               label_file=model + '/test/test_results.pkl')
