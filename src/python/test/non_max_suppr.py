import random

import keras.backend as K
from imageprocessing.Imageprocessing import show
from utils.BoundingBox import BoundingBox
from workdir import work_dir

from src.python.modelzoo.backend.tensor import non_max_suppression
from src.python.modelzoo.models import Yolo
from src.python.utils.fileaccess import VocGenerator

work_dir()

dataset = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                       "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=100).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, 99 - N)
img = batch[idx][0]
model = Yolo(class_names=['gate'])

img_enc = model.preprocessor.encode_img(img)

netout = model.net.predict(img_enc)
boxes = model.postprocessor.decode_netout_to_boxes(netout[0])

sess = K.tf.InteractiveSession()
out = non_max_suppression(boxes, 0.4)
print(out)

sess.close()

boxes_nms = [b for i, b in enumerate(boxes) if i in out]

label_nms = BoundingBox.to_label(boxes_nms)

show(img, labels=label_nms)
