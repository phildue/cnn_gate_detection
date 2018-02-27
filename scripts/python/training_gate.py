# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os
import pprint as pp
import time

import numpy as np

from fileaccess.GateGenerator import GateGenerator
from frontend.models.ssd.SSD import SSD
from frontend.models.yolo.Yolo import Yolo
from workdir import work_dir

work_dir()
from fileaccess.utils import save
from backend.tensor.training import fit_generator

BATCH_SIZE = 8

image_source = "resource/samples/mult_gate_aligned/"
max_epochs = 30
n_samples = 10000
# model = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])
# model = yolo(batch_size=BATCH_SIZE, class_names=['gate'])
predictor = SSD.ssd300(n_classes=1, batch_size=BATCH_SIZE)
# predictor = Yolo.tiny_yolo(class_names=['gate'], batch_size=BATCH_SIZE, color_format='yuv',
#                           weight_file='logs/tinyyolo_25k/TinyYolo.h5')
data_generator = GateGenerator(image_source, batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=n_samples,
                               color_format='yuv')

augmenter = None

model_name = predictor.model.__class__.__name__

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

predictor.preprocessor.augmenter = augmenter

predictor.compile(None)

exp_params = {'model': model_name,
              'resolution': predictor.input_shape,
              'train_params': predictor.model.train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'max_epochs': max_epochs,
              'n_samples': data_generator.n_samples,
              'augmentation': predictor.preprocessor.augmenter.__class__.__name__}

pp.pprint(exp_params)

save(exp_params, 'training_params.txt', result_path, verbose=False)

training_history = fit_generator(predictor, data_generator, out_file=model_name + '.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path, epochs=max_epochs)

save(training_history, 'training_history.pkl', result_path)

os.system('tar -cvf logs/' + name + '.tar.gz ' + result_path)
