from evaluation.evaluation import infer_on_set
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'ewfo_sim',
    'ewfo_voc',
]

datasets = [
    'test_basement_cats',
    'test_basement_gate',
    'test_basement_sign',
    'test_iros_cats',
    'test_iros_gate',
    'test_iros_sign',
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
                             img_res=(416, 416),
                             show_t=1,
                             batch_size=10,
                             model_src=work_dir + model_folder,
                             preprocessing=None,
                             image_source=['resource/ext/samples/{}/'.format(dataset)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
