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

predictor = SSD.ssd300(n_classes=20, batch_size=batch_size, alpha=1.0)
data_generator = VocGenerator(batch_size=batch_size, shuffle=False)

augmenter = YoloAugmenter()

model_name = predictor.net.__class__.__name__

name = model_name + str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

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


training = Training(predictor, data_generator,
                    out_file=model_name + '.h5',
                    patience=-1,
                    log_dir=result_path,
                    stop_on_nan=True,
                    initial_epoch=38,
                    epochs=epochs,
                    log_csv=True,
                    lr_reduce=0.1)

create_dirs([result_path])

pp.pprint(training.summary)

save_file(training.summary, 'training_params.txt', result_path, verbose=False)

training.fit_generator()
