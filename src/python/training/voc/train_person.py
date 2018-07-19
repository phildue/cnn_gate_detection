import argparse
import pprint as pp

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.backend.tensor.yolo.AveragePrecisionYolo import AveragePrecisionYolo
from modelzoo.models.ModelFactory import ModelFactory
from utils.fileaccess.VocGenerator import VocGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

model_name = 'tiny_yolo'
work_dir = 'test'
batch_size = 4
n_samples = None
epochs = 100
initial_epoch = 0
"""
Parse Command Line
"""
cd_work()
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name", default=model_name)
parser.add_argument("--work_dir", help="Working directory", type=str, default=work_dir)
parser.add_argument("--batch_size", help="Batch Size", type=int, default=batch_size)
parser.add_argument("--n_samples", type=int, default=n_samples)
parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--initial_epoch", type=int, default=initial_epoch)
args = parser.parse_args()

model_name = args.model
work_dir = args.work_dir
batch_size = args.batch_size
n_samples = args.n_samples
initial_epoch = args.initial_epoch
"""
Model
"""
augmenter = RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                            (0.5, TransformFlip()),
                            (0.2, RandomShift(-.3, .3))])

predictor = ModelFactory.build(model_name, batch_size)
predictor.preprocessor.augmenter = augmenter

"""
Datasets
"""
train_gen = VocGenerator(batch_size=batch_size, classes=['person'])

"""
Paths
"""
result_path = 'out/' + work_dir + '/'

create_dirs([result_path])

"""
Optimizer Config
"""
params = {'optimizer': 'adam',
          'lr': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}


def average_precision(y_true, y_pred):
    if 'yolo' in model_name:
        return AveragePrecisionYolo(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                    n_classes=predictor.n_classes,
                                    norm=predictor.norm).compute(y_true, y_pred)
    elif 'gate' in model_name:
        return AveragePrecisionGateNet(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                       norm=predictor.norm).compute(y_true, y_pred)
    else:
        raise ValueError("Unknown Model Name for average precision!")


predictor.compile(params=params, metrics=[average_precision])

"""
Training Config
"""

training = Training(predictor, train_gen,
                    out_file='model.h5',
                    patience_early_stop=20,
                    patience_lr_reduce=10,
                    log_dir=result_path,
                    stop_on_nan=True,
                    initial_epoch=initial_epoch,
                    epochs=epochs,
                    log_csv=True,
                    lr_reduce=0.1)

pp.pprint(training.summary)

save_file(training.summary, 'summary.txt', result_path, verbose=False)
predictor.net.backend.summary()

training.fit_generator()