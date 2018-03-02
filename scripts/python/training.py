# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os
import pprint as pp
import time

import numpy as np

from fileaccess.GateGenerator import GateGenerator
from fileaccess.VocGenerator import VocGenerator
from frontend.augmentation.SSDAugmenter import SSDAugmenter
from frontend.models.ssd.SSD import SSD
from workdir import work_dir

work_dir()
from fileaccess.utils import save_file
from backend.tensor.training import fit_generator

BATCH_SIZE = 32

image_source = 'voc'

# model = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])
# model = yolo(batch_size=BATCH_SIZE, class_names=['gate'])
predictor = SSD.ssd300(n_classes=20, batch_size=BATCH_SIZE, alpha=0.1)
data_generator = VocGenerator(batch_size=BATCH_SIZE)

augmenter = SSDAugmenter()

model_name = predictor.net.__class__.__name__

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

predictor.preprocessor.augmenter = augmenter

loss = predictor.loss
predictor.compile(params=None, metrics=[loss.conf_loss_positives,
                                        loss.conf_loss_negatives,
                                        loss.localization_loss]
                  )

exp_params = {'model': model_name,
              'resolution': predictor.img_shape,
              'train_params': predictor.net.train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'n_samples': data_generator.n_samples,
              'augmentation': predictor.preprocessor.augmenter.__class__.__name__}

pp.pprint(exp_params)

save_file(exp_params, 'training_params.txt', result_path, verbose=False)

training_history = fit_generator(predictor, data_generator, out_file=model_name + '.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path, epochs=20)

save_file(training_history, 'training_history.pkl', result_path)
