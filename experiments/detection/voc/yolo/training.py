import os
import sys

from models.Yolo.Yolo import Yolo

os.environ['CUDA_VISIBLE_DEVICES']='1'
PROJECT_ROOT = '/home/philipp/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.VocGenerator import VocSetParser

BATCH_SIZE = 8

model = Yolo(batch_size=BATCH_SIZE)
trainset_generator = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                                  "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=BATCH_SIZE)

params = {'optimizer':'adam'}
model.fit_generator(trainset_generator,BATCH_SIZE,out_file='yolo-voc-adam.h5',params=params)
