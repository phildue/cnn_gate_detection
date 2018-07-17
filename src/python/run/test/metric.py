from modelzoo.evaluation import evaluate_generator, evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.workdir import cd_work

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(500 / batch_size)
image_source = ['resource/ext/samples/daylight_test/']
color_format = 'bgr'

# Model
conf_thresh = 0
summary = load_file('out/1807/'+'baseline104x104-13x13+9layers' + '/summary.pkl')
architecture = summary['architecture']
anchors = summary['anchors']

model = GateNet.create_by_arch(norm=(104,104),
                               architecture=architecture,
                               anchors=anchors,
                               batch_size=batch_size,
                               color_format=color_format,
                               conf_thresh=conf_thresh,
                               augmenter=None,
                               weight_file='out/1807/'+'baseline104x104-13x13+9layers' + '/model.h5'
                               )

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'out/test/'
result_file = 'result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

create_dirs([result_path, result_img_path])
generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=False, color_format=color_format, label_format='xml', start_idx=0,n_samples=500)

evaluate_generator(model, generator, n_batches=n_batches, verbose=True, out_file_labels=result_path + result_file)

evaluate_file(model, result_path + result_file + '.pkl', metrics=[MetricDetection(show_=False,min_box_area=0,max_box_area=104*104)], verbose=True)

