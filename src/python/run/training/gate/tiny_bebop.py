import pprint as pp

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.yolo.AveragePrecisionYolo import AveragePrecisionYolo
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.MavvAugmenter import MavvAugmenter
from utils.workdir import work_dir
import numpy as np
work_dir()

batch_size = 4

image_source = ["resource/ext/samples/bebop20k/"]
max_epochs = 100
n_samples = 20000

anchors = np.array([[0.13809687, 0.27954467],
                    [0.17897748, 0.56287585],
                    [0.36642611, 0.39084195],
                    [0.60043528, 0.67687858]])

predictor = Yolo.tiny_yolo(norm=(160, 320), grid=(5, 10), class_names=['gate'], batch_size=batch_size,
                           color_format='bgr', anchors=anchors)
data_generator = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.1, n_samples=n_samples,
                               color_format='bgr', label_format='xml')

augmenter = MavvAugmenter()

model_name = predictor.net.__class__.__name__

name = 'tiny_bebop'
result_path = 'logs/' + name + '/'

create_dirs([result_path])

predictor.preprocessor.augmenter = augmenter

params = {'optimizer': 'adam',
          'lr': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1e-08,
          'decay': 0.0005}


def average_precision(y_true, y_pred):
    return AveragePrecisionYolo(n_boxes=predictor.n_boxes,
                                grid=predictor.grid,
                                n_classes=1,
                                norm=predictor.norm,
                                batch_size=batch_size).compute(y_true, y_pred)


predictor.compile(params=params, metrics=[average_precision]
                  )


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
                    log_dir=result_path,
                    stop_on_nan=True,
                    initial_epoch=0,
                    epochs=max_epochs,
                    log_csv=True,
                    lr_reduce=0.1,
                    lr_schedule=lr_schedule)

create_dirs([result_path])

pp.pprint(training.summary)

save_file(training.summary, 'training_params.txt', result_path, verbose=False)

training.fit_generator()
