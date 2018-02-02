import os
import sys

from backend.visuals import plot_precision_recall
from evaluation import evaluate_prec_rec_batch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PROJECT_ROOT = '/home/phil/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from models.Yolo.Yolo import Yolo
from fileaccess.VocGenerator import VocSetParser

BATCH_SIZE = 50

set = iter(VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                        "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=BATCH_SIZE).generate())
data = next(set)
model = Yolo(conf_thresh=0, weight_file="dvlab/resource/models/yolo-voc-adam-weights.h5")
interp, meanAP = evaluate_prec_rec_batch(data, model)

plot_precision_recall(interp[1], interp[0], meanAP, output_file='yolo-voc-pr-100.png')
