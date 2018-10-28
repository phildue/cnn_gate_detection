from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

models = [
    'yolov3_w0_416x416',
    'yolov3_w1_416x416',
    'yolov3_w2_416x416',
    'yolov3_w3_416x416',
    'yolov3_arch416x416',
]
datasets = [
    'iros2018_course_final_simple_17gates',
]

work_dir = 'out/thesis/objectdetect/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for d in datasets:
    for model in models:
        for i in range(0, n_iterations):
            model_folder = model + '_i0{}'.format(i)
            prediction_file = 'predictions'.format(d)
            try:
                evalset(name=exp_name,
                        result_path=work_dir + model_folder + '/test_' + d + '/',
                        result_file=prediction_file,
                        batch_size=4,
                        model_src=work_dir + model_folder,
                        preprocessing=None,
                        color_format='bgr',
                        image_source=['resource/ext/samples/{}/'.format(d)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
