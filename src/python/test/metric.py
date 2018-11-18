from modelzoo.models.gatenet.GateNet import GateNet

from evaluation import DetectionEvaluator
from evaluation import evaluate_generator, evaluate_file
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(100 / batch_size)
image_source = ['resource/ext/samples/iros2018course_final_simple/']
color_format = 'bgr'

# Model
conf_thresh = 0
model_dir = 'out/thesis/datagen/' + 'yolov3_gate416x416_i00/'
summary = load_file(model_dir + '/summary.pkl')
architecture = summary['architecture']
anchors = summary['anchors']
img_res = summary['img_res']
model = GateNet.create_by_arch(norm=summary['img_res'],
                               architecture=architecture,
                               anchors=anchors,
                               batch_size=batch_size,
                               color_format=color_format,
                               conf_thresh=conf_thresh,
                               augmenter=None,
                               weight_file=model_dir + '/model.h5'
                               )

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'out/test/'
result_file = 'result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'
ObjectLabel.classes = ['gate']
create_dirs([result_path, result_img_path])
generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=False, color_format=color_format, label_format='xml', start_idx=0, n_samples=500)

evaluate_generator(model, generator, n_batches=n_batches, verbose=True, out_file_labels=result_path + result_file)

evaluate_file(model, result_path + result_file + '.pkl',
              metrics=[DetectionEvaluator(show_=True, min_box_area=0, max_box_area=img_res[0] * img_res[1])], verbose=True)
