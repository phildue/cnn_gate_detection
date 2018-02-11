# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os
import pprint as pp
import time

import numpy as np

from augmentation.SSDAugmenter import SSDAugmenter
from fileaccess.VocGenerator import VocGenerator
from models.ssd.SSD import SSD
from workdir import work_dir

work_dir()
from fileaccess.utils import save
from backend.training import fit_generator

BATCH_SIZE = 8

image_source = "resource/voc_train_val/VOC2012/JPEGImages/"

# model = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])
# model = yolo(batch_size=BATCH_SIZE, class_names=['gate'])
predictor = SSD.ssd7(n_classes=20)
data_generator = VocGenerator("resource/voc_train_val/VOC2012/Annotations/",
                              image_source, batch_size=8,
                              valid_frac=0.1)
augmenter = SSDAugmenter()

model_name = predictor.model.__class__.__name__

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

predictor.preprocessor.augmenter = augmenter

predictor.compile(None)

exp_params = {'model': model_name,
              'resolution': predictor.img_shape,
              'train_params': predictor.model.train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'n_samples': data_generator.n_samples,
              'augmentation': predictor.preprocessor.augmenter.__class__.__name__}

pp.pprint(exp_params)

save(exp_params, 'training_params.txt', result_path, verbose=False)

training_history = fit_generator(predictor, data_generator, out_file=model_name + '.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path, epochs=20)

save(training_history, 'training_history.pkl', result_path)

os.system('tar -cvf logs/' + name + '.tar.gz ' + result_path)
