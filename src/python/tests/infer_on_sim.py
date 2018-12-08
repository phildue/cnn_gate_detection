from evaluation.evaluation import infer_on_set
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'mavlabgates',
    # 'mavnet',
    # 'mavnet_lowres160',
    # 'mavnet_lowres320',
    # 'mavnet_strides',
    # 'mavnet_strides3_pool2',
    # 'mavnet_strides4_pool1',
    # 'yolov3_width0',
    # 'yolov3_width1',
    # 'yolov3_width2',
    # 'yolov3_width3',
    # 'yolo_lowres160',
]
preprocessing = [
    # [TransformResize((80, 80))],
    # None,
    # [TransformResize((160, 160))],
    # [TransformResize((320, 320))],
    # [TransformResize((320, 320))],
    # [TransformResize((320, 320))],
    # [TransformResize((320, 320))],
    # None,
    # None,
    # None,
    # None,
    # [TransformResize((160, 160))],
]

img_res = [
    (80,80),
    # (416, 416),
    # (160, 160),
    # (320, 320),
    # (320, 320),
    # (320, 320),
    # (320, 320),
    # (160, 160),
    # (416, 416),
    # (416, 416),
    # (416, 416),
    # (416, 416),
    # (160, 160),
    # [TransformResize((160, 160))],
]
dataset = 'iros2018_course_final_simple_17gates'

work_dir = 'out/'
n_iterations = 2
ObjectLabel.classes = ['gate']
for i_m, model in enumerate(models):
    for i in range(0, n_iterations):
        model_folder = model + '_i0{}'.format(i)
        try:
            infer_on_set(result_path=work_dir + model_folder + '/test_' + dataset + '/',
                         result_file='predictions',
                         img_res=img_res[i_m],
                         batch_size=4,
                         model_src=work_dir + model_folder,
                         preprocessing=preprocessing[i_m],
                         image_source=['resource/ext/samples/{}/'.format(dataset)])
        except FileNotFoundError:
            print("Not found: {}".format(model_folder))
