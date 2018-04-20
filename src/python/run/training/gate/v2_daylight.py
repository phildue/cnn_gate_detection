import pprint as pp

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.callbacks.MeanAveragePrecision import MeanAveragePrecision
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

cd_work()

"""
Model
"""
batch_size = 4
augmenter = RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                            (0.5, TransformFlip()),
                            (0.2, RandomShift(-.3, .3))])

predictor = Yolo.yolo_v2(class_names=['gate'], batch_size=batch_size, color_format='bgr')
predictor.preprocessor.augmenter = augmenter
pp.pprint(predictor.net.backend.summary())

"""
Datasets
"""
image_source = ["resource/ext/samples/daylight/"]
test_image_source_1 = ['resource/ext/samples/industrial_new_test/']
test_image_source_2 = ['resource/ext/samples/daylight_test/']

train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.1,
                          color_format='bgr', label_format='xml')
test_gen_1 = GateGenerator(test_image_source_1, batch_size=batch_size, valid_frac=0, color_format='bgr',
                           label_format='xml')
test_gen_2 = GateGenerator(test_image_source_2, batch_size=batch_size, valid_frac=0, color_format='bgr',
                           label_format='xml')

"""
Paths
"""
result_path = 'logs/v2_daylight/'
test_result_1 = result_path + 'results/industrial--'
test_result_2 = result_path + 'results/daylight--'

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

save_file(training.summary, 'training_params.txt', result_path, verbose=False)

training.fit_generator()
