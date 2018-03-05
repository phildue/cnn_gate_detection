import keras.backend as K
from backend.tensor.iou import non_max_suppression_tf
from frontend.utils.BoundingBox import BoundingBox, centroid_to_minmax
from imageprocessing.Backend import resize
from imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from workdir import work_dir

from src.python.modelzoo.backend.tensor.metrics.AveragePrecision import AveragePrecision
from src.python.modelzoo.models.yolo.Yolo import Yolo
from src.python.utils.fileaccess import GateGenerator

work_dir()
conf_thresh = 0.1
generator = GateGenerator(directory='resource/samples/mult_gate_aligned_test/', batch_size=100, color_format='bgr',
                          shuffle=False, start_idx=500)

# generator = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
#                          "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=8, color_format='bgr')

# model = SSD.ssd300(n_classes=20, weight_file='logs/ssd300/SSD300.h5', conf_thresh=0.01, color_format='bgr')
model = Yolo.yolo_v2(class_names=['gate'], conf_thresh=conf_thresh, color_format='yuv',
                     weight_file='logs/yolov2_25k/YoloV2.h5')
batch = next(generator.generate())
idx = 56

img_height, img_width = 416, 416
label_true = batch[idx][1]
img = batch[idx][0]

label_pred_2 = model.predict(img)

img, label_true = resize(img, label=label_true, shape=(416, 416))

mAP = AveragePrecision(n_boxes=845, iou_thresh=0.4)

img_enc = model.preprocessor.encode_img(img)
label_pred_raw_t = model.net.predict(img_enc)[0]

label_true_t = model.preprocessor.encode_label(label_true)
#
# label_true_t = K.get_session().run(
#     YoloLoss.assign_anchors(K.np.expand_dims(label_true_t, 0), K.np.expand_dims(label_pred_raw_t, 0)))[0]
coord_true_t = label_true_t[:, :, :, :4]
coord_true_t = model.postprocessor.decode_coord(coord_true_t)
class_true_t = label_true_t[:, :, :, 5:] * K.np.expand_dims(label_true_t[:, :, :, 4], -1)
class_true_t[:, :, 1:, :] = 0

coord_true_t = K.np.reshape(coord_true_t, (-1, 4))
class_true_t = K.np.reshape(class_true_t, (-1, 1))

coord_pred_t = label_pred_raw_t[:, :, :, :4]
coord_pred_t = model.postprocessor.decode_coord(coord_pred_t)
class_pred_t = label_pred_raw_t[:, :, :, 5:]
conf_pred_t = label_pred_raw_t[:, :, :, 4]

class_pred_t = K.np.reshape(class_pred_t, (-1, 1))

coord_pred_t = K.np.reshape(coord_pred_t, (-1, 4))
conf_pred_t = (K.np.reshape(conf_pred_t, (-1, 1))).flatten()

idx = K.get_session().run(non_max_suppression_tf(coord_pred_t, conf_pred_t, conf_thresh))

for i in range(coord_pred_t.shape[0]):
    if i not in idx:
        class_pred_t[i, :] = 0
        conf_pred_t[i] = 0

boxes_pred = BoundingBox.from_tensor_centroid(class_pred_t, coord_pred_t, conf_pred_t)
boxes_pred = [b for b in boxes_pred if b.c > conf_thresh]
label_pred = BoundingBox.to_label(boxes_pred)

show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])

coord_true_t = centroid_to_minmax(coord_true_t)
coord_pred_t = centroid_to_minmax(coord_pred_t)
coord_true_t = K.constant(coord_true_t)
class_true_t = K.constant(class_true_t)
coord_pred_t = K.constant(coord_pred_t)
class_pred_t = K.constant(class_pred_t)

with K.get_session() as sess:
    detections = mAP.detections(coord_true_t, coord_pred_t, class_true_t, class_pred_t, conf_thresh=conf_thresh)
    sess.run(K.tf.global_variables_initializer())
    tp, fp, fn = sess.run(detections)
    print("True positives ", tp)
    print("False positives ", fp)
    print("False negatives ", fn)
    sess.run(K.tf.global_variables_initializer())

show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])

# precision, recall = sess.run(
#     mAP.precision_recall(coord_true_t, coord_pred_t, class_true_t, class_pred_t, conf_thresh))
# print("Precision", precision)
# print("Recall", recall)

# average_precision = K.get_session().run(mAP.average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t))
# print("Average Precision", average_precision)
# show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
# label_pred = ssd.postprocessor.decode_label(label_pred_t[0])

# img = annotate_text("Loss: " + str(loss[0]), img, thickness=2, color=(0, 0, 0))
