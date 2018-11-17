import os
import random
import sys

from imageprocessing.Backend import resize, show, annotate_bounding_box
from src.python.modelzoo.backend.tensor import YoloV2
from src.python.modelzoo.backend.tensor import metrics_tensor

from src.python.modelzoo.models import Yolo

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from src.python.utils.fileaccess import VocGenerator
import keras.backend as K

dataset = VocGenerator("resource/samples/VOCdevkit/VOC2012/Annotations/",
                       "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=100).generate()

conf_thresh = 0.3
model = Yolo(conf_thresh=conf_thresh)
while True:
    batch = next(dataset)
    idx = random.randint(0, 100)

    img, label = batch[idx]
    img, label = resize(img, shape=(416, 416), label=label)
    y_true = model.preprocessor.encode_label(label)
    img_pp = model.preprocessor.encode_img(img)
    netout = model.net.predict(img_pp)

    sess = K.tf.InteractiveSession()
    K.tf.initialize_all_variables()
    y_pred_k = K.constant(netout, shape=(1, 13, 13, 5, 25))
    y_true_k = K.constant(y_true, name="y_true", shape=(1, 13, 13, 125))
    y_true_k = K.reshape(y_true_k, (1, 13, 13, 5, 25))
    y_true_raw = K.reshape(y_true_k, (1, 13, 13, 5, 25))
    y_true_filtered = YoloV2.match_true_boxes_tensor(y_true_raw,
                                                     YoloV2._get_iou(y_pred_k[:, :, :, :, :4],
                                                                     y_true_k[:, :, :, :, :4]))

    tp, fn, fp = metrics_tensor(y_pred_k, y_true_filtered)

    print("True positives = " + str(tp.eval()))
    print("False positives = " + str(fp.eval()))
    print("False negatives = " + str(fn.eval()))

    # precision_ = precision(tp, fp)
    # recall_ = sess.eval(recall(tp, fn))
    # print(precision_)
    # print(recall_)
    # show(img_pred, "Decoded prediction")
    # show(img_truth, "Decoded truth")
    img_ann = annotate_bounding_box(img, model.postprocessor.decode_netout_to_label(y_pred_k[0].eval()), (255, 0, 0))
    img_ann = annotate_bounding_box(img_ann, label)
    show(img_ann, "Out ")
    sess.close()
