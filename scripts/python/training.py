# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os

import time

import numpy as np
from models.ssd.SSD import SSD

from workdir import work_dir
import pprint as pp

work_dir()
from fileaccess.GateGenerator import GateGenerator
from models.yolo.Yolo import Yolo
from fileaccess.utils import save
from backend.training import fit_generator
from models.yolo.TinyYolo import TinyYolo

BATCH_SIZE = 8

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'
image_source = 'resource/samples/mult_gate'
augmenter = None

# model = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])
# model = yolo(batch_size=BATCH_SIZE, class_names=['gate'])
predictor = SSD((416, 416, 3), 1)
if not os.path.exists(result_path):
    os.makedirs(result_path)

model_name = predictor.model.__class__.__name__

predictor.preprocessor.augmenter = augmenter

data_generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, valid_frac=0.1)

predictor.compile(None)

exp_params = {'model': model_name,
              'train_params': predictor.model.train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'n_samples': data_generator.n_samples,
              'augmentation': predictor.preprocessor.augmenter.__class__.__name__}

pp.pprint(exp_params)

save(exp_params, 'training_params.txt', result_path, verbose=False)

training_history = fit_generator(predictor, data_generator, out_file=model_name + '.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path)

save(training_history, 'training_history.pkl', result_path)

os.system('tar -cvf logs/' + name + '.tar.gz ' + result_path)
