from modelzoo.evaluation import evaluate_generator, evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(500 / batch_size)
image_source = ['resource/ext/samples/industrial_new_test/']
color_format = 'bgr'

# Model
conf_thresh = 0

model = ModelFactory.build('GateNetV10', batch_size=batch_size)

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'out/test/'
result_file = 'result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

create_dirs([result_path, result_img_path])
generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=False, color_format=color_format, label_format='xml', start_idx=0)

evaluate_generator(model, generator, n_batches=n_batches, verbose=True, out_file_labels=result_path + result_file)

#evaluate_file(model, result_path + result_file + '.pkl', metrics=[MetricDetection(show_=False)], verbose=True,
#              out_file_metric=result_path + result_file + '.pkl')

