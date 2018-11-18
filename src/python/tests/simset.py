from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'mavnet',
    'mavnet_lowres160',
    'mavnet_lowres320',
    'mavnet_strides',
    'mavnet_strides3_pool2',
    'mavnet_strides4_pool1',
]
dataset = 'iros2018_course_final_simple_17gates'


work_dir = 'out/'
n_iterations = 1
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for model in models:
    for i in range(0, n_iterations):
        model_folder = model + '_i0{}'.format(i)
        prediction_file = 'predictions'.format(dataset)
        try:
            evalset(name=exp_name,
                    result_path=work_dir + model_folder + '/test_' + dataset + '/',
                    result_file=prediction_file,
                    batch_size=4,
                    model_src=work_dir + model_folder,
                    preprocessing=None,
                    color_format='bgr',
                    image_source=['resource/ext/samples/{}/'.format(dataset)])
        except FileNotFoundError:
            print("Not found: {}".format(model_folder))
