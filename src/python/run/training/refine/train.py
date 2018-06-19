import argparse
import pprint as pp

import numpy as np

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

model_name = 'GateNet3x3V3'
work_dir = 'gate_3x3V3'
batch_size = 4
n_samples = None
epochs = 100
initial_epoch = 0
learning_rate = 0.001


# Why do we need such a big network? Because we have to deal with overlaps and scales
# If we look at a smaller crop there will be less possible combinations, hence we can make that network much simpler
# 52x52 crop over layer/width
def learning_rate_schedule(epoch):
    if epoch > 50:
        return 0.0001
    else:
        return 0.001


model_src = None
img_res = (52, 52)

cd_work()
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name",
                    type=str, default=model_name)
parser.add_argument("--work_dir", help="Working directory", type=str, default=work_dir)
parser.add_argument("--image_source", help="List of folders to be scanned for train images",
                    default=["resource/ext/samples/daylight/","resource/ext/samples/industrial_new/"])
parser.add_argument("--test_image_source_1", help="List of folders to be scanned for test images", type=str,
                    default='resource/ext/samples/industrial_new_test/')
parser.add_argument("--test_image_source_2", help="List of folders to be scanned for test images", type=str,
                    default='resource/ext/samples/daylight_test/')
parser.add_argument("--batch_size", help="Batch Size", type=int, default=batch_size)
parser.add_argument("--n_samples", type=int, default=n_samples)
parser.add_argument("--initial_epoch", type=int, default=initial_epoch)
parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--learning_rate", type=float, default=learning_rate)
parser.add_argument("--model_src", default=model_src)
parser.add_argument("--img_width", default=img_res[1], type=int)
parser.add_argument("--img_height", default=img_res[0], type=int)

args = parser.parse_args()

model_name = args.model_name
work_dir = args.work_dir
batch_size = args.batch_size
n_samples = args.n_samples
initial_epoch = args.initial_epoch
learning_rate = args.learning_rate
model_src = args.model_src
epochs = args.epochs
image_source = args.image_source
test_image_source_1 = [args.test_image_source_1]
test_image_source_2 = [args.test_image_source_2]
img_res = args.img_height, args.img_width

"""
Model
"""
augmenter = None#RandomEnsemble([(1.0, RandomBrightness(0.8, 1.2)),
                 #           (0.5, TransformFlip()),
                  #          (0.2, RandomShift(-0.2,0.2))])

predictor = GateNet.create(model_name, batch_size=batch_size, norm=img_res, grid=[(6, 6)], scale_coor=5.0,
                           anchors=np.array([[[1, 1],
                                              [0.3, 0.3],
                                              [2, 1],
                                              [1, 0.5],
                                              [0.7, 0.7]
                                              ]]
                                            ))
predictor.preprocessor.augmenter = augmenter

"""
Datasets
"""

train_gen = CropGenerator(GateGenerator(image_source, batch_size=batch_size, valid_frac=0.05,
                                        color_format='bgr', label_format='xml', n_samples=n_samples))
# test_gen_1 = GateGenerator(test_image_source_1, batch_size=batch_size, valid_frac=0, color_format='bgr',
#                           label_format='xml')
# test_gen_2 = GateGenerator(test_image_source_2, batch_size=batch_size, valid_frac=0, color_format='bgr',
#                          label_format='xml')

"""
Paths
"""
result_path = 'out/' + work_dir + '/'
# test_result_1 = result_path + 'results/set_1-'
# test_result_2 = result_path + 'results/set_2--'

# create_dirs([result_path, result_path + '/results/'])

"""
Optimizer Config
"""
params = {'optimizer': 'adam',
          'lr': learning_rate,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}


def average_precision(y_true, y_pred):
    return AveragePrecisionGateNet(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                   norm=predictor.norm).compute(y_true, y_pred)


predictor.compile(params=params,metrics=[average_precision])

"""
Training Config
"""
#
# test_metric_1 = MeanAveragePrecision(predictor=predictor,
#                                      test_set=test_gen_1,
#                                      out_file=test_result_1,
#                                      period=5)
#
# test_metric_2 = MeanAveragePrecision(predictor=predictor,
#                                      test_set=test_gen_2,
#                                      out_file=test_result_2,
#                                      period=5)

training = Training(predictor, train_gen,
                    out_file='model.h5',
                    patience_early_stop=20,
                    patience_lr_reduce=10,
                    log_dir=result_path,
                    stop_on_nan=True,
                    lr_schedule=learning_rate_schedule,
                    initial_epoch=initial_epoch,
                    epochs=epochs,
                    log_csv=True,
                    lr_reduce=0.1)

pp.pprint(training.summary)

save_file(training.summary, 'summary.txt', result_path, verbose=False)
predictor.net.backend.summary()

training.fit_generator()
