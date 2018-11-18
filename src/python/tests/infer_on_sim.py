from evaluation.evalset import infer_on_set
from utils.imageprocessing.transform.TransformCrop import TransformCrop
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    # 'mavnet',
    # 'mavnet_lowres160',
    'mavnet_lowres320',
    # 'mavnet_strides',
    # 'mavnet_strides3_pool2',
    # 'mavnet_strides4_pool1',
    # 'yolov3_width0',

]
preprocessing = [
    # [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    # [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))],
    # [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))],
    # None,
]
dataset = 'iros2018_course_final_simple_17gates'

work_dir = 'out/'
n_iterations = 1
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for i_m, model in enumerate(models):
    for i in range(0, n_iterations):
        model_folder = model  # + '_i0{}'.format(i)
        try:
            infer_on_set(result_path=work_dir + model_folder + '/test_' + dataset + '/',
                         result_file='predictions',
                         batch_size=4,
                         model_src=work_dir + model_folder,
                         preprocessing=preprocessing[i_m],
                         color_format='bgr',
                         image_source=['resource/ext/samples/{}/'.format(dataset)])
        except FileNotFoundError:
            print("Not found: {}".format(model_folder))
