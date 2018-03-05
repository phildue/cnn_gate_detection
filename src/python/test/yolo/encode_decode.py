import random

import keras.backend as K
from imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from workdir import work_dir

from src.python.modelzoo.models.yolo.Yolo import Yolo
from src.python.utils.fileaccess import VocGenerator

work_dir()
batch_size = 1000
dataset = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                       "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=batch_size).generate()
batch = next(dataset)
N = 10
idx = random.sample(range(0, batch_size), N)

yolo = Yolo.tiny_yolo()
labels1_enc = []
for j in idx:
    label1_t = yolo.preprocessor.encoder.encode_label(batch[j][1])
    label1_t = K.np.expand_dims(label1_t, 0)
    labels1_enc.append(label1_t)
labels1_t = K.np.concatenate(labels1_enc)

for i, j in enumerate(idx):
    img = batch[j][0]
    label_true = batch[j][1]
    label_dec = yolo.postprocessor.decoder.decode_netout_to_label(labels1_t[i])
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
