from evaluation.evaluation import infer_on_set
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'ewfo_sim',
    'mavnet_random_view',
    'mavnet_race_court',
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
                         img_res=(416, 416),
                         batch_size=4,
                         model_src=work_dir + model_folder,
                         preprocessing=None,
                         image_source=['resource/ext/samples/{}/'.format(dataset)])
        except FileNotFoundError:
            print("Not found: {}".format(model_folder))
