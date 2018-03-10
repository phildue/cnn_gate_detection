import pprint as pp
import time

import numpy as np
from modelzoo.backend.tensor.Training import Training

from modelzoo.augmentation.YoloAugmenter import YoloAugmenter
from modelzoo.models.ssd.SSD import SSD
from utils.fileaccess.VocGenerator import VocGenerator
from utils.fileaccess.utils import save_file, create_dirs
from utils.workdir import work_dir

work_dir()

batch_size = 32

image_source = 'voc'
work_path = 'logs/ssd300_voc/'

predictor = SSD.ssd300(n_classes=20, batch_size=batch_size, alpha=1.0, weight_file=work_path + '/SSD300.h5')
data_generator = VocGenerator(batch_size=batch_size, shuffle=True)

augmenter = YoloAugmenter()

model_name = predictor.net.__class__.__name__

epochs = 120  # ~40 000 iterations
train_params = {'optimizer': 'SGD',
                'lr': 0.001,
                'momentum': 0.9,
                'epsilon': 1e-08,
                'decay': 0.0005}

loss = predictor.loss
predictor.compile(params=train_params, metrics=[loss.conf_loss_positives,
                                                loss.conf_loss_negatives,
                                                loss.localization_loss]
                  )
predictor.preprocessor.augmenter = augmenter


def lr_schedule(epoch):
    if 0 <= epoch < 80:
        return 0.001
    elif 80 <= epoch <= 100:
        return 0.0001
    else:
        return 0.000001


training = Training(predictor, data_generator,
                    out_file=model_name + '.h5',
                    patience=-1,
                    log_dir=work_path,
                    stop_on_nan=True,
                    initial_epoch=49,
                    epochs=epochs,
                    log_csv=True,
                    lr_reduce=0.1,
                    lr_schedule=lr_schedule)

create_dirs([work_path])

pp.pprint(training.summary)

save_file(training.summary, 'summary.txt', work_path, verbose=False)

training.fit_generator()
