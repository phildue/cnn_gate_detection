import os
import sys

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from models.Yolo.Yolo import Yolo
from fileaccess.GateGenerator import GateGenerator
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/stream_valid', batch_size=8, n_samples=100, color_format='bgr',
                          shuffle=False, start_idx=30)

model = Yolo(class_names=['gate'], weight_file='logs/yolo-gate-mult-05/yolo-gate-adam.h5', conf_thresh=0.3)

demo_generator(model, generator, t_show=1)
