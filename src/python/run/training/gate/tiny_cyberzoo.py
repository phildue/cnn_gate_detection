import pprint as pp
import time

import numpy as np

from modelzoo.backend.tensor.Training import Training
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.augmentation.AugmenterDistort import AugmenterDistort
from utils.imageprocessing.augmentation.AugmenterEnsemble import AugmenterEnsemble
from utils.workdir import work_dir

work_dir()

BATCH_SIZE = 2

image_source = ["resource/samples/bebop/"]
max_epochs = 4
n_samples = 20
dist_model_file = 'resource/barrel_dist_model.pkl'

predictor = Yolo.tiny_yolo(norm=(80, 166), grid=(2, 5), class_names=['gate'], batch_size=BATCH_SIZE,
                           color_format='bgr')
data_generator = GateGenerator(image_source, batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=n_samples,
                               color_format='bgr', label_format='xml')

augmenter = AugmenterEnsemble(augmenters=[(1.0, AugmenterDistort(BarrelDistortion.from_file(dist_model_file)))])
# TODO put in Brightness/Contrast augmentation etc
# TODO check how yolo does normalization
model_name = predictor.net.__class__.__name__

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

create_dirs([result_path])

predictor.preprocessor.augmenter = augmenter

params = {'optimizer': 'adam',
          'lr': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}

loss = predictor.loss
predictor.compile(params, metrics=[loss.localization_loss, loss.confidence_loss])

training = Training(predictor, data_generator,
                    out_file=model_name + '.h5',
                    patience=5,
                    log_dir=result_path,
                    stop_on_nan=True,
                    initial_epoch=0,
                    epochs=max_epochs,
                    log_csv=True,
                    lr_reduce=0.1)

create_dirs([result_path])

pp.pprint(training.summary)

save_file(training.summary, 'training_params.txt', result_path, verbose=False)

training.fit_generator()
