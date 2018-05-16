import argparse
import pprint as pp

from modelzoo.backend.tensor.ModelConverter import ModelConverter
from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.callbacks.MeanAveragePrecision import MeanAveragePrecision
from modelzoo.models.ModelBuilder import ModelBuilder
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

cd_work()
parser = argparse.ArgumentParser()
parser.add_argument("model", help="model name",
                    type=str)
parser.add_argument("work_dir", help="Working directory", type=str)
parser.add_argument("--image_source", help="List of folders to be scanned for train images", type=str,
                    default="resource/ext/samples/mixed_rooms/")
parser.add_argument("--test_image_source_1", help="List of folders to be scanned for test images", type=str,
                    default='resource/ext/samples/industrial_new_test/')
parser.add_argument("--test_image_source_2", help="List of folders to be scanned for test images", type=str,
                    default='resource/ext/samples/daylight_test/')
parser.add_argument("--batch_size", help="Batch Size", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=None)

args = parser.parse_args()

"""
Model
"""
batch_size = args.batch_size
augmenter = RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                            (0.5, TransformFlip()),
                            (0.2, RandomShift(-.3, .3))])

predictor = ModelBuilder.get_model(args.model, batch_size)
predictor.preprocessor.augmenter = augmenter

"""
Datasets
"""
image_source = [args.image_source]
test_image_source_1 = [args.test_image_source_1]
test_image_source_2 = [args.test_image_source_2]

train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.05,
                          color_format='bgr', label_format='xml', n_samples=args.n_samples)
test_gen_1 = GateGenerator(test_image_source_1, batch_size=batch_size, valid_frac=0, color_format='bgr',
                           label_format='xml')
test_gen_2 = GateGenerator(test_image_source_2, batch_size=batch_size, valid_frac=0, color_format='bgr',
                           label_format='xml')

"""
Paths
"""
result_path = 'out/' + args.work_dir + '/'
test_result_1 = result_path + 'results/set_1-'
test_result_2 = result_path + 'results/set_2--'

create_dirs([result_path, result_path + '/results/'])

"""
Optimizer Config
"""
params = {'optimizer': 'adam',
          'lr': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}

predictor.compile(params=params)

"""
Training Config
"""

test_metric_1 = MeanAveragePrecision(predictor=predictor,
                                     test_set=test_gen_1,
                                     out_file=test_result_1)

test_metric_2 = MeanAveragePrecision(predictor=predictor,
                                     test_set=test_gen_2,
                                     out_file=test_result_2)

training = Training(predictor, train_gen,
                    out_file='model.h5',
                    patience_early_stop=20,
                    patience_lr_reduce=10,
                    log_dir=result_path,
                    stop_on_nan=True,
                    initial_epoch=0,
                    epochs=100,
                    log_csv=True,
                    lr_reduce=0.1,
                    callbacks=[test_metric_1, test_metric_2])

pp.pprint(training.summary)

save_file(training.summary, 'summary.txt', result_path, verbose=False)
predictor.net.backend.summary()

training.fit_generator()

"""
Create TFLite Model
"""

ModelConverter(args.model, result_path).finalize()
