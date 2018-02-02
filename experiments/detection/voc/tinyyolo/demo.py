import glob
import os
import sys

from imageprocessing.Backend import annotate_bounding_box, resize, show
from models.Yolo.TinyYolo import TinyYolo

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.VocGenerator import VocSetParser

dataset = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                       "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=2).generate()
sample_files = glob.glob('resource/samples/VOCdevkit/VOC2012/JPEGImages/*.jpg')
model_myimpl = TinyYolo(model_file='dvlab/resource/models/tiny-yolo-voc-myimpl-adam.h5',conf_thresh=0.5)
model_yadk = TinyYolo(model_file='dvlab/resource/models/tiny-yolo-voc-yad2k.h5')

for batch in dataset:
    for sample, label in batch:
        label_pred_my = model_myimpl.predict(sample)
        label_pred_yadk = model_yadk.predict(sample)
        img, label = resize(sample, shape=(416, 416),label=label)
        img = annotate_bounding_box(img, label_pred_my, (0, 0, 255))
        img = annotate_bounding_box(img, label_pred_yadk, (255, 0, 0))
        img = annotate_bounding_box(img, label, (0, 255, 0))
        show(img, 'Merged')

