import os
import sys
from os.path import expanduser

PROJECT_ROOT = expanduser('~') + '/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from models.Yolo.TinyYolo import TinyYolo
from fileaccess.GateGenerator import GateGenerator
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/mult_gate', batch_size=8, n_samples=100, color_format='yuv')

model = TinyYolo(class_names=['gate'], weight_file='logs/tiny-yolo-gate-mult-2/tiny-yolo-gate-adam.h5', conf_thresh=0.1)

demo_generator(model, generator)
