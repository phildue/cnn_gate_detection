import numpy as np

from modelzoo.models.ssd.SSD import SSD
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show, LEGEND_BOX
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import work_dir

work_dir()

predictor = SSD.ssd300(conf_thresh=0, neg_min=0.1)

dummy = ObjectLabel('object', np.zeros((4, 2)))
dummy.y_min = 50
dummy.y_max = 100
dummy.x_min = 100
dummy.x_max = 200
true_label = ImgLabel([dummy])
anchor_boxes_t = predictor.encoder.encode_label_batch([ImgLabel([])])
label_t = predictor.encoder.encode_label_batch([true_label])

anchor_boxes = predictor.decoder.decode_netout_to_boxes(anchor_boxes_t[0], -1)
label = predictor.decoder.decode_netout_to_boxes(label_t[0], -1)

responsible_boxes = []
unvalid_boxes = []
background_boxes = []
for i in range(len(anchor_boxes)):
    if label[i].c > 0:
        responsible_boxes.append(anchor_boxes[i])
    elif label_t[0, i, 0] == 1:
        background_boxes.append(anchor_boxes[i])
    else:
        unvalid_boxes.append(anchor_boxes[i])

black_image = Image(np.zeros(predictor.input_shape), 'bgr')

responsible_label = BoundingBox.to_label(responsible_boxes)
unvalid_label = BoundingBox.to_label(unvalid_boxes)
background_label = BoundingBox.to_label(background_boxes)
show(img=black_image, labels=[responsible_label, true_label], colors=[(255, 255, 255), (0, 255, 0)], legend=LEGEND_BOX,
     name='responsible')
show(img=black_image, labels=[unvalid_label, true_label], colors=[(255, 0, 255), (0, 255, 0)], legend=LEGEND_BOX,
     name='unvalid')
show(img=black_image, labels=[background_label, true_label], colors=[(255, 255, 0), (0, 255, 0)], legend=LEGEND_BOX,
     name='background')

for i in range(len(anchor_boxes)):
    label = BoundingBox.to_label(anchor_boxes[i:i + 5])
    show(img=black_image, labels=label, legend=LEGEND_BOX, t=1)
