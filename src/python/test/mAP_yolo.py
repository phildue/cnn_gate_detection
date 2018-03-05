import keras.backend as K
from backend.tensor.yolo.AveragePrecisionYolo import AveragePrecisionYolo
from frontend.utils.BoundingBox import BoundingBox
from imageprocessing.Backend import resize
from imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED, COLOR_BLUE
from workdir import work_dir

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
idx = 2

img_height, img_width = 416, 416
label_true = batch[idx][1]
img = batch[idx][0]

label_pred_2 = model.predict(img)

img, label_true = resize(img, label=label_true, shape=(416, 416))

mAP = AveragePrecisionYolo(n_boxes=5, iou_thresh=0.4, n_classes=1)

img_enc = model.preprocessor.encode_img(img)
label_pred_t = model.net.predict(img_enc)[0]

label_true_t = model.preprocessor.encode_label(label_true)

show(img, labels=[label_true, label_pred_2], colors=[COLOR_GREEN, COLOR_BLUE])

with K.get_session() as sess:
    detections = mAP.detections(label_true_t, label_pred_t, conf_thresh=conf_thresh)
    sess.run(K.tf.global_variables_initializer())
    tp, fp, fn = sess.run(detections)
    print("True positives ", tp)
    print("False positives ", fp)
    print("False negatives ", fn)
    sess.run(K.tf.global_variables_initializer())

    boxes_pred = model.postprocessor.postprocess(label_pred_t)
    label_pred = BoundingBox.to_label(boxes_pred)
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
