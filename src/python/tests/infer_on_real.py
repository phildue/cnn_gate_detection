from evaluation.evaluation import infer_on_set
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
datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]

preprocessing = [
    # [TransformResize((120, 160))],
    [TransformResize((240, 320))],
    # [TransformResize((240, 320))],
    # [TransformResize((240, 320))],
    # [TransformCrop(80, 0, 640 - 80, 480), TransformResize((416, 416))],

]
work_dir = 'out/'
n_iterations = 1
ObjectLabel.classes = ['gate']
for i_m, model in enumerate(models):
    for dataset in datasets:
        for i in range(0, n_iterations):
            model_folder = model  # + '_i0{}'.format(i)
            try:
                infer_on_set(result_path=work_dir + model_folder + '/test_' + dataset + '/',
                             result_file='predictions',
                             batch_size=4,
                             model_src=work_dir + model_folder,
                             preprocessing=preprocessing[i_m],
                             image_source=['resource/ext/samples/{}/'.format(dataset)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
