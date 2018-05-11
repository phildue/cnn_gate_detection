from modelzoo.evaluation.SpeedEvaluator import SpeedEvaluator
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

cd_work()

name = 'speed'
work_dir = 'out/gatev8_mixed/'
"""
Image Source
"""
batch_size = 20
n_batches = 6
image_source = ['resource/ext/samples/daylight_test']
color_format = 'bgr'

generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=True, color_format=color_format, label_format='xml')

"""
Model config
"""
conf_thresh = 0.3
weight_file = work_dir + 'model.h5'
model = GateNet.v5(conf_thresh=conf_thresh, batch_size=batch_size)

"""
Evaluator
"""
result_path = work_dir + name + '/'
result_file = 'result.pkl'
exp_param_file = 'experiment_parameters.txt'

create_dirs([result_path])

evaluator = SpeedEvaluator(model,
                           out_file=result_path + result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'evaluator': evaluator.__class__.__name__,
              'conf_thresh': conf_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * batch_size}

save_file(exp_params, exp_param_file, result_path)
