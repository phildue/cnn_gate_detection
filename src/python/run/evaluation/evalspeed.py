import argparse

from modelzoo.evaluation.SpeedEvaluator import SpeedEvaluator
from modelzoo.models.ModelBuilder import ModelBuilder
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model name",
                    type=str)
parser.add_argument("work_dir", help="Working directory", type=str)
parser.add_argument("--image_source", help="List of folders to be scanned for test images", type=str,
                    default='resource/ext/samples/daylight_test')
parser.add_argument("--batch_size", help="Batch Size", type=int, default=1)
parser.add_argument("--n_batches", help="Amount of batches", type=int, default=None)
parser.add_argument("--result_file_name", type=str, default='speed_result.pkl')
args = parser.parse_args()

cd_work()

name = 'speed'
work_dir = args.work_dir
"""
Image Source
"""
batch_size = args.batch_size
image_source = [args.image_source]
color_format = 'bgr'

generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=True, color_format=color_format, label_format='xml')
n_batches = args.n_batches if args.n_batches is not None else int(generator.n_samples / batch_size)

"""
Model config
"""
model = ModelBuilder.build(args.model, args.batch_size)
"""
Evaluator
"""
result_path = work_dir + name + '/'
result_file = args.result_file_name
exp_param_file = 'experiment_parameters.txt'

create_dirs([result_path])

evaluator = SpeedEvaluator(model, out_file=result_path + result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'evaluator': evaluator.__class__.__name__,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * batch_size}

save_file(exp_params, exp_param_file, result_path)
