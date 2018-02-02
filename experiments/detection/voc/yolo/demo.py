import glob
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
from fileaccess.VocGenerator import VocSetParser

dataset = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                       "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=10).generate()
sample_files = glob.glob('resource/samples/VOCdevkit/VOC2012/JPEGImages/*.jpg')
model_myimpl = Yolo(weight_file="yolo-voc-adam-weights.h5", conf_thresh=0.3)

for batch in dataset:
    for sample, label in batch:
        label_pred_my = model_myimpl.predict_show(sample, label)
