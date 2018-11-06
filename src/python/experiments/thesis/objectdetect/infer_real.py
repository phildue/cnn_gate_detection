from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]

models = [
    # 'objectdetect/yolov3_w01_416x416',
    # 'datagen/yolov3_grid416x416',
    # 'datagen/yolov3_pool416x416',
    'objectdetect/yolov3_d3_416x416',
    # 'objectdetect/yolov3_w0_416x416',
    # 'objectdetect/yolov3_w1_416x416',
    # 'objectdetect/yolov3_w2_416x416',
    # 'objectdetect/yolov3_w3_416x416',
    # 'objectdetect/yolov3_d02_416x416',
    # 'objectdetect/yolov3_d01_416x416',
    # 'objectdetect/yolov3_d0_416x416',
    # 'objectdetect/yolov3_d1_416x416',
    # 'objectdetect/yolov3_d2_416x416',
    # 'datagen/yolov3_arch2416x416',
]
datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]

work_dir = 'out/thesis/'
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
                        img_res=(480, 640),
                        result_path=work_dir + model_folder + '/test_' + d + '/',
                        result_file=prediction_file,
                        batch_size=4,
                        # n_samples=40,
                        model_src=work_dir + model_folder,
                        preprocessing=None,
                        color_format='bgr',
                        image_source=['resource/ext/samples/{}/'.format(d)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
