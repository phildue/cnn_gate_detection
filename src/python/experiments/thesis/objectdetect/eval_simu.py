from modelzoo.evaluation.evalmetric import evalmetric
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]

models = [
    'objectdetect/yolov3_w01_416x416',
    'datagen/yolov3_grid416x416',
    'objectdetect/yolov3_pool416x416',
    'objectdetect/yolov3_avg_pool416x416',
    'objectdetect/yolov3_k9_416x416',
    'objectdetect/yolov3_w0_416x416',
    'objectdetect/yolov3_w1_416x416',
    'objectdetect/yolov3_w2_416x416',
    'objectdetect/yolov3_w3_416x416',
    'objectdetect/yolov3_d02_416x416',
    'objectdetect/yolov3_d01_416x416',
    'objectdetect/yolov3_d0_416x416',
    'objectdetect/yolov3_d1_416x416',
    'objectdetect/yolov3_d2_416x416',
    'objectdetect/yolov3_d3_416x416',
    'datagen/yolov3_arch2416x416',
]
datasets = [
    'iros2018_course_final_simple_17gates',
]

work_dir = 'out/thesis/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for d in datasets:
    for model in models:
        for i in range(0, n_iterations):
            model_folder = model + '_i0{}'.format(i)
            prediction_file = 'predictions_{}'.format(d, i)
            label_file = work_dir + model_folder + '/test_' + d + '/' + 'predictions.pkl'
            for iou_thresh in [0.4, 0.6, 0.8]:
                try:
                    evalmetric(name='results_iou{}'.format(iou_thresh),
                               min_box_area=0.01,
                               max_box_area=2.0,
                               min_aspect_ratio=0.33,
                               max_aspect_ratio=3.0,
                               iou_thresh=iou_thresh,
                               model_src=work_dir + model_folder,
                               label_file=label_file,
                               result_path=work_dir + model_folder + '/test_' + d + '/',
                               show=False)
                except FileNotFoundError:
                    print("Missing: " + label_file)
                    continue
