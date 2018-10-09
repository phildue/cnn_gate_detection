from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    'yolov3_gate_realbg416x416',
    'yolov3_gate416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_uniform416x416',
    'yolov3_gate_mixed416x416',
    'yolov3_pp416x416'

]
datasets = [
    # 'real_test_labeled',
    # 'jevois_cyberzoo',
    # 'jevois_basement',
    # 'jevois_hallway',
    'iros2018_course_final_simple_17gates'
]

box_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
work_dir = 'out/thesis/datagen/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'scenegen'
for d in datasets:
    for model in models:
        for i in range(n_iterations):
            model_folder = model + '_i0{}'.format(i)
            prediction_file = 'predictions_{}'.format(d, i)
            min_box_area = box_sizes[0]
            max_box_area = box_sizes[-1]
            label_file = work_dir + model_folder + '/test_' + d + '/' + 'predictions.pkl'
            for iou_thresh in [0.4, 0.6, 0.8]:
                try :
                    evalmetric(name='results_{}_boxes{}-{}_iou{}'.format(d, min_box_area, max_box_area, iou_thresh, i),
                               min_box_area=min_box_area,
                               max_box_area=max_box_area,
                               min_aspect_ratio=0.3,
                               max_aspect_ratio=4.0,
                               iou_thresh=iou_thresh,
                               batch_size=16,
                               model_src=work_dir + model_folder,
                               color_format='bgr',
                               label_file=label_file,
                               result_path=work_dir + model_folder + '/' + d + '/',
                               show=False)
                except FileNotFoundError:
                    print("Missing: " + label_file)
                    continue
                #
                # for i_box_size in range(len(box_sizes) - 1):
                #     min_box_area = box_sizes[i_box_size]
                #     max_box_area = box_sizes[i_box_size + 1]
                #     evalmetric(name='results_{}_boxes{}-{}_iou{}'.format(d, min_box_area, max_box_area, iou_thresh, i),
                #                min_box_area=min_box_area,
                #                max_box_area=max_box_area,
                #                min_aspect_ratio=0.3,
                #                max_aspect_ratio=4.0,
                #                iou_thresh=iou_thresh,
                #                batch_size=16,
                #                model_src=work_dir + model_folder,
                #                color_format='bgr',
                #                label_file=work_dir + model_folder + '/' + exp_name + '/' + prediction_file + '.pkl',
                #                result_path=work_dir + model_folder + '/' + exp_name + '/',
                #                show=False)