import keras.backend as K
from utils.BoundingBox import BoundingBox

from modelzoo.models.yolo import AveragePrecisionYolo, DetectionCountYolo
from modelzoo.models.yolo import PrecisionRecallYolo
from modelzoo.models.yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.timing import toc, tic, tuc
from utils.workdir import cd_work

cd_work()
conf_thresh = 0.1
batch_size = 10
tic()
generator = GateGenerator(directories=['resource/samples/mult_gate_aligned_test/'], batch_size=batch_size,
                          color_format='bgr', label_format='xml',
                          shuffle=False, start_idx=900)
toc("Files found in ")

tic()
model = Yolo.yolo_v2(class_names=['gate'], conf_thresh=conf_thresh, color_format='yuv', batch_size=batch_size)
model.preprocessor.augmenter = None
toc("Model created in ")

tic()
batch = next(generator.generate())
toc("Images loaded in ")

tic()
img_t, label_true_t = model.preprocessor.preprocess_train(batch)
toc("Images preprocessed in ")

tic()
label_pred_t = model.net.predict(img_t)
toc("Predicted in ")

tic()
average_precision = AveragePrecisionYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)
precision_recall = PrecisionRecallYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)
detection_count = DetectionCountYolo(n_boxes=5, iou_thresh=0.4, n_classes=1, batch_size=batch_size)

y_true = K.placeholder(shape=label_true_t.shape, dtype=K.tf.float64)
y_pred = K.placeholder(shape=label_true_t.shape, dtype=K.tf.float64)
pr = precision_recall.compute(y_true, y_pred)
tuc()
ap = average_precision.compute(y_true, y_pred)
tuc()
dc = detection_count.compute(y_true, y_pred)
toc("Graph created in ")

for i in range(batch_size):
    label_true = batch[i][1]
    img = batch[i][0]
    img, label_true = resize(img, label=label_true, shape=model.input_shape)
    show(img, labels=[label_true], colors=[COLOR_GREEN], t=1)

with K.get_session() as sess:
    sess.run(K.tf.global_variables_initializer())

    tic()
    tp, fp, fn = sess.run(dc, {y_true: label_true_t,
                               y_pred: label_pred_t})
    toc("Detections in ")

    tic()
    precision, recall, n_predictions = sess.run(pr, {y_true: label_true_t,
                                                     y_pred: label_pred_t})
    toc("Precision Recall in ")

    tic()
    ap_out = sess.run(ap, {y_true: label_true_t,
                           y_pred: label_pred_t})
    toc("Average Precision in ")
    boxes_pred = model.postprocessor.response2sample(label_pred_t)

for i in range(batch_size):
    print("True positives ", tp[i])
    print("False positives ", fp[i])
    print("False negatives ", fn[i])
    print("Precision", precision[i])
    print("Recall", recall[i])
    print("Predictions:", n_predictions[i])
    print("Ap", ap_out[i])
    label_pred = BoundingBox.to_label(boxes_pred[i])
    label_true = batch[i][1]
    img = batch[i][0]
    img, label_true = resize(img, label=label_true, shape=model.input_shape)
    show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
