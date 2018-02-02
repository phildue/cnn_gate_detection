import os
import sys
from os.path import expanduser

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

PROJECT_ROOT = expanduser('~') + '/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from models.Yolo.TinyYolo import TinyYolo
from fileaccess.GateGenerator import GateGenerator
from fileaccess.utils import save
from backend.training import fit_generator

result_path = 'logs/tiny-yolo-gate-mult-2/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

BATCH_SIZE = 8

yolo = TinyYolo(batch_size=BATCH_SIZE, class_names=['gate'])

trainset_gen = GateGenerator(directory='resource/samples/mult_gate/', batch_size=BATCH_SIZE, valid_frac=0.1, n_samples=32)
train_params = {'optimizer': 'adam',
                'lr': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.0005}

yolo.compile(train_params)

training_history = fit_generator(yolo, trainset_gen, out_file='yolo-gate-adam.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0, log_dir=result_path)
save(training_history, 'training_history.pkl', result_path)
