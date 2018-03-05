import random

import keras.backend as K
from imageprocessing.Backend import annotate_text
from imageprocessing.Imageprocessing import COLOR_RED, COLOR_GREEN, show
from workdir import work_dir

from src.python.modelzoo.models import Yolo
from src.python.utils.fileaccess import VocGenerator

work_dir()

dataset = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                       "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=100).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, 99 - N)

model = Yolo()
# img_enc = ssd.preprocessor.encode_img(img)
labels1_enc = []
labels2_enc = []
for i in range(N):
    label1_t = model.preprocessor.encode_label(batch[i][1])
    label1_t = K.np.expand_dims(label1_t, 0)
    labels1_enc.append(label1_t)
    label2_t = model.preprocessor.encode_label(batch[i + 5][1])
    label2_t = K.np.expand_dims(label2_t, 0)
    labels2_enc.append(label2_t)

labels1_t = K.np.concatenate(labels1_enc, 0)
labels2_t = K.np.concatenate(labels2_enc, 0)
# label_dec = ssd.postprocessor.decode_label(label_enc)
# label_enc_2 = ssd.preprocessor.encode_label(label_2)
sess = K.tf.InteractiveSession()
loss = model.loss(y_pred=labels2_t, y_true=labels1_t)
print("Total Loss:", K.np.sum(loss))

for i in range(N):
    img = batch[i][0]
    label_true = batch[i][1]
    label_pred = batch[i + 5][1]
    img = annotate_text(str(loss[i]), img, thickness=2, color=(0, 0, 0))
    show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
