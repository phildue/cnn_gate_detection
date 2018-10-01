from modelzoo.evaluation.evalmetric import evalmetric
from modelzoo.evaluation.evalset import evalset
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = []

work_dir = 'out/thesis/datagen/'
n_iterations = 2
ObjectLabel.classes = ['gate']
dataset = 'random_iros'
for i in range(n_iterations):
    model_folder = 'yolov3_gate_uniform416x416_i0{}'.format(i)
    # evalset(name='complex_vs_efo',
    #         result_path=work_dir + model_folder + '/' + dataset + '/',
    #         result_file='predictions.pkl',
    #         batch_size=16,
    #         model_src=work_dir + model_folder,
    #         preprocessing=None,
    #         color_format='bgr',
    #         image_source=['resource/ext/samples/' + dataset + '/'])

    # for iou_thresh in [0.4, 0.6, 0.8]:
    #     evalmetric(name='detections_iou{}'.format(iou_thresh),
    #                min_box_area=0.01,
    #                max_box_area=2.0,
    #                min_aspect_ratio=.3,
    #                max_aspect_ratio=3.0,
    #                iou_thresh=iou_thresh,
    #                batch_size=16,
    #                model_src=work_dir + model_folder,
    #                color_format='bgr',
    #                label_file=work_dir + model_folder + '/'+dataset+'/predictions.pkl',
    #                result_path=work_dir + model_folder + '/' + dataset + '/',
    #                show=False)

ObjectLabel.classes = ['gate']
n_iterations = 1

dataset = 'iros2018_cats'
for i in range(n_iterations):
    model_folder = 'yolov3_cats_uniform416x416_i0{}'.format(i)
    evalset(name='complex_vs_efo',
            result_path=work_dir + model_folder + '/' + dataset + '/',
            result_file='predictions.pkl',
            batch_size=16,
            model_src=work_dir + model_folder,
            preprocessing=None,
            color_format='bgr',
            image_source=['resource/ext/samples/' + dataset + '/'])

    for iou_thresh in [0.4, 0.6, 0.8]:
        evalmetric(name='detections_iou{}'.format(iou_thresh),
                   min_box_area=0.01,
                   max_box_area=2.0,
                   min_aspect_ratio=.3,
                   max_aspect_ratio=3.0,
                   iou_thresh=iou_thresh,
                   batch_size=16,
                   model_src=work_dir + model_folder,
                   color_format='bgr',
                   label_file=work_dir + model_folder + '/'+dataset+'/predictions.pkl',
                   result_path=work_dir + model_folder + '/' + dataset + '/',
                   show=False)