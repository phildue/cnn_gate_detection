import random

import keras.backend as K

from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import annotate_text, resize
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.workdir import cd_work
import numpy as np

cd_work()
batch_size = 10
img_res = 416, 416
predictor = Yolo.create_by_arch(norm=img_res,
                                anchors=np.array([[[81, 82],
                                                   [135, 169],
                                                   [344, 319]],
                                                  [[10, 14],
                                                   [23, 27],
                                                   [37, 58]]]),
                                architecture=[
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'max_pool', 'size': (2, 2)},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'max_pool', 'size': (2, 2)},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'max_pool', 'size': (2, 2)},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'max_pool', 'size': (2, 2)},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'max_pool', 'size': (2, 2)},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 1024, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 256, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'predict'},
                                    {'name': 'route', 'index': [-4]},
                                    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 128, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'upsample', 'size': 2},
                                    {'name': 'route', 'index': [-1, 8]},
                                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
                                     'alpha': 0.1},
                                    {'name': 'predict'}], color_format='bgr', augmenter=None, class_names=[
        "gate"
    ])

dataset = GateGenerator(["resource/ext/samples/daylight15k/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=99).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, N)

# img_enc = ssd.preprocessor.encode_img(img)
labels1_enc = []
labels2_enc = []
for i in range(N):
    label1_t = predictor.encoder.encode_label(batch[i][1])
    label1_t = K.np.expand_dims(label1_t, 0)
    labels1_enc.append(label1_t)
    label2_t = predictor.encoder.encode_label(batch[i + 1][1])
    label2_t = K.np.expand_dims(label2_t, 0)
    labels2_enc.append(label2_t)

labels1_t = K.np.concatenate(labels1_enc, 0)
labels2_t = K.np.concatenate(labels2_enc, 0)
# label_dec = ssd.postprocessor.decode_label(label_enc)
# label_enc_2 = ssd.preprocessor.encode_label(label_2)
sess = K.tf.InteractiveSession()
loss = predictor.loss.compute(y_pred=K.constant(labels2_t, K.tf.float32),
                              y_true=K.constant(labels1_t, K.tf.float32)).eval()
print("Total Loss:", K.np.sum(loss))

for i in range(N):
    img = batch[i][0]
    label_true = batch[i][1]
    label_pred = batch[i + 1][1]
    img_res, label_true = resize(img, (416, 416), label=label_true)
    _, label_pred = resize(img, (416, 416), label=label_pred)
    img_res = annotate_text(str(loss[0, i]), img_res, thickness=2, color=(255, 255, 255))
    show(img_res, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
