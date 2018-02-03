from models.Yolo.Yolo import Yolo
from fileaccess.VocGenerator import VocSetParser

BATCH_SIZE = 50

set = iter(VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                        "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=BATCH_SIZE).generate())
data = next(set)
model = Yolo(conf_thresh=0, weight_file="dvlab/resource/models/yolo-voc-adam-weights.h5")
interp, meanAP = evaluate_prec_rec_batch(data, model)

plot_precision_recall(interp[1], interp[0], meanAP, output_file='yolo-voc-pr-100.png')
