import keras.backend as K
from modelzoo.backend.tensor.yolo.AveragePrecisionYolo import AveragePrecisionYolo
from modelzoo.backend.tensor.yolo.DetectionCountYolo import DetectionCountYolo
from modelzoo.backend.tensor.yolo.PrecisionRecallYolo import PrecisionRecallYolo

from modelzoo.models.yolo.Yolo import Yolo
from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.workdir import work_dir

work_dir()
conf_thresh = 0.1
batch_size = 5
generator = GateGenerator(directories=['resource/samples/mult_gate_aligned_test/'], batch_size=batch_size,
                          color_format='bgr', label_format='xml',
                          shuffle=False, start_idx=900)

# generator = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
#                          "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=8, color_format='bgr')

# model = SSD.ssd300(n_classes=20, weight_file='logs/ssd300/SSD300.h5', conf_thresh=0.01, color_format='bgr')
model = Yolo.yolo_v2(class_names=['gate'], conf_thresh=conf_thresh, color_format='yuv', batch_size=batch_size)
batch = next(generator.generate())
model.preprocessor.augmenter = None

average_precision = AveragePrecisionYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)
precision_recall = PrecisionRecallYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)
detection_count = DetectionCountYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)

img_t, label_true_t = model.preprocessor.preprocess_train(batch)
label_pred_t = model.net.predict(img_t)

with K.get_session() as sess:
    for i in range(batch_size):
        label_true = batch[i][1]
        img = batch[i][0]
        img, label_true = resize(img, label=label_true, shape=(416, 416))
        show(img, labels=[label_true], colors=[COLOR_GREEN])

    pr = precision_recall.compute(label_true_t, label_pred_t)
    ap = average_precision.compute(label_true_t, label_pred_t)
    dc = detection_count.compute(label_true_t, label_pred_t)
    sess.run(K.tf.global_variables_initializer())
    precision, recall, n_predictions = sess.run(pr)
    ap_out = sess.run(ap)
    dc_out = sess.run(dc)

    for i in range(batch_size):
        # print("True positives ", tp[:, i])
        # print("False positives ", fp[:, i])
        # print("False negatives ", fn[:, i])
        print("Precision", precision[i, :])
        print("Recall", recall[i, :])
        print("Predictions:", n_predictions[i, :])
        print("Ap", ap_out[i])
        boxes_pred = model.postprocessor.postprocess(label_pred_t[i])
        label_pred = BoundingBox.to_label(boxes_pred)
        label_true = batch[i][1]
        img = batch[i][0]
        img, label_true = resize(img, label=label_true, shape=(416, 416))
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
