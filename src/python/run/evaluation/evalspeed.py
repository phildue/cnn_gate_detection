import argparse

from modelzoo.evaluation.SpeedEvaluator import SpeedEvaluator
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model name",
                    type=str)
parser.add_argument("work_dir", help="Working directory", type=str)
parser.add_argument("--image_source", help="List of folders to be scanned for test images", type=[str],
                    default='resource/ext/samples/daylight_test')
parser.add_argument("--batch_size", help="Batch Size", type=int, default=1)
parser.add_argument("--n_batches", help="Amount of batches", type=int, default=None)
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
if args.model == "tiny_yolo":
    model = Yolo.tiny_yolo(batch_size=batch_size)
elif args.model == "yolo":
    model = Yolo.yolo_v2(batch_size=batch_size)
elif args.model == "gatev5":
    model = GateNet.v5(batch_size=batch_size)
elif args.model == "gatev6":
    model = GateNet.v6(batch_size=batch_size)
elif args.model == "gatev7":
    model = GateNet.v7(batch_size=batch_size)
elif args.model == "gatev8":
    model = GateNet.v8(batch_size=batch_size)
elif args.model == "gatev9":
    model = GateNet.v9(batch_size=batch_size)
elif args.model == "gatev10":
    model = GateNet.v10(batch_size=batch_size)
elif args.model == "gatev11":
    model = GateNet.v11(batch_size=batch_size)
elif args.model == "gatev12":
    model = GateNet.v12(batch_size=batch_size)
elif args.model == "gatev13":
    model = GateNet.v13(batch_size=batch_size)
else:
    raise ValueError("Unknown model name!")
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
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * batch_size}

save_file(exp_params, exp_param_file, result_path)
