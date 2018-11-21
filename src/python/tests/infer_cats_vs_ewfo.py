from evaluation.evaluation import infer_on_set
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'mavnet',
    'yolov3_width2',
    'cats_deep',
    'cats'
    'sign'
]
preprocessing = [
    None,
    None,
    None,
    None,
]

img_res = [
    (416, 416),
    (416, 416),
    (416, 416),
    (416, 416),
]
datasets = [
    'test_basement_cats',
    'test_basement_gate',
    'test_basement_sign',
]

work_dir = 'out/'
n_iterations = 2
ObjectLabel.classes = ['gate']
for i_d, dataset in enumerate(datasets):
    for i_m, model in enumerate(models):
        for i in range(0, n_iterations):
            model_folder = model + '_i0{}'.format(i)
            try:
                infer_on_set(result_path=work_dir + model_folder + '/test_' + dataset + '/',
                             result_file='predictions',
                             img_res=img_res[i_m],
                             batch_size=10,
                             model_src=work_dir + model_folder,
                             preprocessing=preprocessing[i_m],
                             color_format='bgr',
                             image_source=['resource/ext/samples/{}/'.format(dataset)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
