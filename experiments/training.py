# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os

import time

import numpy as np

from workdir import work_dir

work_dir()
from fileaccess.GateGenerator import GateGenerator
from models.Yolo.Yolo import Yolo
from fileaccess.utils import save
from backend.training import fit_generator
from models.Yolo.TinyYolo import TinyYolo

BATCH_SIZE = 8

name = str(int(np.round(time.time() / 1000)))
result_path = 'logs/' + name + '/'
image_source = 'resource/samples/mult_gate_train'
augment_active = False
train_params = {'optimizer': 'adam',
                'lr': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.0005}

model = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])
# model = Yolo(batch_size=BATCH_SIZE, class_names=['gate'])

if not os.path.exists(result_path):
    os.makedirs(result_path)

model.preprocessor.augment_active = augment_active

trainset = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=32)

model.compile(train_params)

training_history = fit_generator(model, trainset, out_file=result_path + 'yolo-gate-adam.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0)
save(training_history, 'training_history.pkl', result_path)

exp_params = {'model': model.__class__.__name__,
              'train_params': train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'n_samples': 1,
              'augmentation': augment_active}

save(exp_params, 'training_params.pkl', result_path)

os.system('tar -cvf logs/' + name + '.tar.gz ' + result_path)
