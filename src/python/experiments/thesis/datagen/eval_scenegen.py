from modelzoo.evaluation.evalmetric import evalmetric
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    # 'yolov3_gate_realbg416x416',
    # 'yolov3_gate416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_gate_varioussim416x416',
    # 'yolov3_gate_uniform416x416',
    # 'yolov3_gate_mixed416x416',
    # 'yolov3_pp416x416'

]
datasets = [
    # 'real_test_labeled',
    # 'jevois_cyberzoo',
    # 'jevois_basement',
    # 'jevois_hallway',
    # 'iros2018_course_final_simple_17gates',
    'iros_nocats'
    # 'basement_white100'
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1
ObjectLabel.classes = ['gate']
exp_name = 'scenegen'
for d in datasets:
    for model in models:
        for i in range(n_iterations):
            model_folder = model + '_i0{}'.format(i)
            prediction_file = 'predictions_{}'.format(d, i)
            label_file = work_dir + model_folder + '/test_' + d + '/' + 'predictions.pkl'
            for iou_thresh in [0.4, 0.6, 0.8]:
                try:
                    evalmetric(name='results_iou{}'.format(iou_thresh),
                               min_box_area=0,
                               max_box_area=100.0,
                               min_aspect_ratio=0,
                               max_aspect_ratio=100.0,
                               iou_thresh=iou_thresh,
                               model_src=work_dir + model_folder,
                               label_file=label_file,
                               result_path=work_dir + model_folder + '/test_' + d + '/',
                               show=False)
                except FileNotFoundError:
                    print("Missing: " + label_file)
                    continue

