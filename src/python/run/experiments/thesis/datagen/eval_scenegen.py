from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    # 'yolov3_gate_realbg416x416',
    # 'yolov3_gate416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_uniform416x416',
    'yolov3_gate_mixed416x416'
]
datasets = [
    'real_test_labeled',
    # 'jevois_cyberzoo',
    # 'jevois_basement',
    # 'jevois_hallway',
    # 'iros2018_course_final_simple_17gates'
]
work_dir = 'out/thesis/datagen/'
n_iterations = 1
ObjectLabel.classes = ['gate']
for d in datasets:
    for model in models:
        for i in range(n_iterations):
            model_folder = model + '_i0{}'.format(i)
            evalset(name='scenegen',
                    result_path=work_dir + model_folder + '/test/',
                    result_file='scenegen_predictions_{}.pkl'.format(d),
                    batch_size=16,
                    model_src=work_dir + model_folder,
                    preprocessing=None,
                    color_format='bgr',
                    image_source=['resource/ext/samples/{}/'.format(d)])

            for iou_thresh in [0.4, 0.6, 0.8]:
                evalmetric(name='scenegen_results_{}_iou{}'.format(d, iou_thresh),
                           min_box_area=0,
                           max_box_area=1.2,
                           iou_thresh=iou_thresh,
                           batch_size=16,
                           model_src=work_dir + model_folder,
                           color_format='bgr',
                           label_file=work_dir + model_folder + '/test/scenegen_predictions_{}.pkl'.format(d),
                           result_path=work_dir + model_folder + '/test/',
                           show=False)
