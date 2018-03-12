# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pprint as pp
import time

import numpy as np
from modelzoo.augmentation.AugmenterEnsemble import AugmenterEnsemble

from modelzoo.backend.tensor.Training import Training
from modelzoo.models.ssd.SSD import SSD
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.TransformSubsample import TransformSubsample
from utils.workdir import work_dir

work_dir()

BATCH_SIZE = 2

image_source = ["resource/samples/mult_gate_aligned/"]
max_epochs = 4
n_samples = 20
predictor = SSD.ssd7(n_classes=1, batch_size=BATCH_SIZE, color_format='yuv')
# predictor = Yolo.tiny_yolo(norm=(208, 208), grid=(6, 6), class_names=['gate'], batch_size=BATCH_SIZE,
#                            color_format='yuv')
data_generator = GateGenerator(image_source, batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=n_samples,
                               color_format='yuv', label_format='xml')

augmenter = AugmenterEnsemble(augmenters=[(0.5, TransformSubsample())])

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
predictor.compile(params, metrics=[loss.localization_loss, loss.conf_loss_positives, loss.conf_loss_negatives])

training = Training(predictor, data_generator,
                    out_file=model_name + '.h5',
                    patience=3,
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
