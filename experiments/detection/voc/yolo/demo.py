import glob

from models.Yolo.Yolo import Yolo
from fileaccess.VocGenerator import VocSetParser

dataset = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                       "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=10).generate()
sample_files = glob.glob('resource/samples/VOCdevkit/VOC2012/JPEGImages/*.jpg')
model_myimpl = Yolo(weight_file="yolo-voc-adam-weights.h5", conf_thresh=0.3)

for batch in dataset:
    for sample, label in batch:
        label_pred_my = model_myimpl.predict_show(sample, label)
