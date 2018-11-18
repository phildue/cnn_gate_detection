from evaluation import evalmetric
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
model = 'yolov3_cats_uniform416x416'

dataset = 'basement_white100_cats'


work_dir = 'out/thesis/datagen/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'scenegen'
for i in range(n_iterations):
    model_folder = model + '_i0{}'.format(i)
    label_file = work_dir + model_folder + '/test_' + dataset + '/' + 'predictions.pkl'
    for iou_thresh in [0.4, 0.6, 0.8]:
        try :
            evalmetric(name='results_iou{}'.format(iou_thresh),
                       min_box_area=0,
                       max_box_area=2.0,
                       min_aspect_ratio=0.3,
                       max_aspect_ratio=4.0,
                       iou_thresh=iou_thresh,
                       model_src=work_dir + model_folder,
                       label_file=label_file,
                       result_path=work_dir + model_folder + '/test_' + dataset + '/',
                       show=False)
        except FileNotFoundError:
            print("Missing: " + label_file)
            continue
