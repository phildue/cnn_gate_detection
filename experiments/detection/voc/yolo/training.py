from fileaccess.VocGenerator import VocSetParser
from models.Yolo.Yolo import Yolo

BATCH_SIZE = 8

model = Yolo(batch_size=BATCH_SIZE)
trainset_generator = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                                  "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=BATCH_SIZE)

params = {'optimizer':'adam'}
model.fit_generator(trainset_generator,BATCH_SIZE,out_file='yolo-voc-adam.h5',params=params)
