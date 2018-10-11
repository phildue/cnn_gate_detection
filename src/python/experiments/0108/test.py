from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = ['mavnet208x208-13x13+9layers',
          'mavnet_dense208x208-13x13+9layers',
          'mavnet_multiscale_dense208x208',
          'mavnet_multiscale_dense_compression208x208']

work_dir = 'out/0108/'

for model in models:
    preprocessing = None if 'gray' not in model else TransformGray()
    evalset(name='rf',
            result_path=work_dir + model + '/test/',
            result_file='daylight_industrial.pkl',
            batch_size=24,
            # n_samples=48,
            model_src=work_dir + model,
            preprocessing=None,
            color_format='yuv',
            image_source=['resource/ext/samples/daylight_test/', 'resource/ext/samples/industrial_new_test/'])

box_range = [0.0, 1.0]
for model in models:
    for iou_thresh in [0.4, 0.6]:
        for i in range(len(box_range) - 1):
            evalmetric(name='total_iou{}'.format(iou_thresh, box_range[i]),
                       min_box_area=box_range[i],
                       max_box_area=box_range[i + 1],
                       iou_thresh=iou_thresh,
                       model_src=work_dir + model,
                       label_file=work_dir + model + '/test/daylight_industrial.pkl',
                       result_path=work_dir + model + '/test/',
                       show=False)

box_range = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
for model in models:
    for iou_thresh in [0.4, 0.6]:
        for i in range(len(box_range) - 1):
            evalmetric(name='total_iou{}_range{}-{}'.format(iou_thresh, box_range[i], box_range[i + 1]),
                       min_box_area=box_range[i],
                       max_box_area=box_range[i + 1],
                       iou_thresh=iou_thresh,
                       model_src=work_dir + model,
                       label_file=work_dir + model + '/test/daylight_industrial.pkl',
                       result_path=work_dir + model + '/test/',
                       show=False)
