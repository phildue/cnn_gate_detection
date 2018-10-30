from modelzoo.evaluation.evalmetric import evalmetric
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

models = [
    'yolov3_gate_realbg416x416',
    'yolov3_gate_uniform416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_mixed416x416',
    'yolov3_allgen416x416',
    'yolov3_gate416x416',
    'yolov3_hsv416x416',
    'yolov3_blur416x416',
    'yolov3_chromatic416x416',
    'yolov3_exposure416x416',
    'yolov3_40k416x416',
    'yolov3_allview416x416',
    'yolov3_arch2416x416',

]
datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for d in datasets:
    for model in models:
        for i in range(0,n_iterations):
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
                    print("Not found: {}".format(model_folder))
