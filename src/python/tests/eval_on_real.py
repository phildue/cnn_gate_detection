from evaluation.evaluation import evaluate_labels
from utils.imageprocessing.transform.TransformCrop import TransformCrop
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]

models = [
    # 'mavnet',
    # 'mavnet_lowres160',
    'mavnet_lowres320',
    # 'mavnet_strides',
    # 'mavnet_strides3_pool2',
    # 'mavnet_strides4_pool1',
    'yolov3_width0',

]

preprocessing = [
    # [TransformResize((120, 160))],
    [TransformResize((240, 320))],
    # [TransformResize((240, 320))],
    # [TransformResize((240, 320))],
    [TransformCrop(80, 0, 640 - 80, 480), TransformResize((416, 416))],

]

resolution = [
    (120, 160),
    (240, 320),
    (240, 320),
    (240, 320),
]

datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]
work_dir = 'out/'
n_iterations = 1
ObjectLabel.classes = ['gate']
for i_m, model in enumerate(models):
    for dataset in datasets:
        for i in range(0, n_iterations):
            model_folder = model  # +'_i0{}'.format(i)
            label_file = work_dir + model_folder + '/test_' + dataset + '/' + 'predictions.pkl'
            img_size = resolution[i_m][0] * resolution[i_m][1]
            for iou_thresh in [0.4, 0.6, 0.8]:
                try:
                    evaluate_labels(name='results_iou{}'.format(iou_thresh),
                                    preprocessing=preprocessing[i_m],
                                    min_box_area=0.01 * img_size,
                                    max_box_area=2.0 * img_size,
                                    min_aspect_ratio=0.33,
                                    max_aspect_ratio=3.0,
                                    iou_thresh=iou_thresh,
                                    model_src=work_dir + model_folder,
                                    label_file=label_file,
                                    result_path=work_dir + model_folder + '/test_' + dataset + '/',
                                    show=1)
                except FileNotFoundError:
                    print("Missing: " + label_file)
                    continue
