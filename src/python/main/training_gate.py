# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pprint as pp
import time

import numpy as np
from frontend.augmentation.AugmenterPixel import AugmenterPixel
from frontend.models.ssd.SSD import SSD
from workdir import work_dir

from src.python.modelzoo.augmentation.AugmenterEnsemble import AugmenterEnsemble
from src.python.utils.fileaccess import GateGenerator

work_dir()
from src.python.utils.fileaccess import save_file, create_dir
from src.python.modelzoo.backend.tensor.training import fit_generator

BATCH_SIZE = 32

image_source = ["resource/samples/mult_gate_aligned_blur_distort/", "resource/samples/mult_gate_aligned/"]
max_epochs = 60
n_samples = 20000
predictor = SSD.ssd7(n_classes=1, batch_size=BATCH_SIZE, color_format='yuv')
# predictor = Yolo.tiny_yolo(norm=(208, 208), grid=(6, 6), class_names=['gate'], batch_size=BATCH_SIZE,
#                            color_format='yuv')
data_generator = GateGenerator(image_source, batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=n_samples,
                               color_format='yuv')

augmenter = AugmenterEnsemble(augmenters=[(0.5, AugmenterPixel())])

model_name = predictor.net.__class__.__name__

name = str(int(np.round(time.time() / 10)))
result_path = 'logs/' + name + '/'

create_dir([result_path])

predictor.preprocessor.augmenter = augmenter

params = {'optimizer': 'adam',
          'lr': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}

loss = predictor.loss
predictor.compile(params, metrics=[loss.localization_loss, loss.conf_loss_positives, loss.conf_loss_negatives])

exp_params = {'model': model_name,
              'resolution': predictor.input_shape,
              'train_params': predictor.net.train_params,
              'image_source': image_source,
              'batch_size': BATCH_SIZE,
              'max_epochs': max_epochs,
              'n_samples': data_generator.n_samples,
              'augmentation': predictor.preprocessor.augmenter.__class__.__name__}

pp.pprint(exp_params)

save_file(exp_params, 'training_params.txt', result_path, verbose=False)

training_history = fit_generator(predictor, data_generator, out_file=model_name + '.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path, epochs=max_epochs)

save_file(training_history, 'training_history.pkl', result_path)

