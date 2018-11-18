import keras.backend as K
import numpy as np
from modelzoo.ModelFactory import ModelFactory

from evaluation import DetectionEvaluator
from evaluation import DetectionResult
from modelzoo.metrics.DetectionCountGateNet import DetectionCountGateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_BLUE
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(500 / batch_size)
image_source = ['resource/ext/samples/industrial_new_test/']
color_format = 'bgr'

# Model
conf_thresh = 0

model = ModelFactory.build('GateNetV10', batch_size=batch_size)

# Evaluator
iou_thresh = 0.4

# Result Paths
generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=True, color_format=color_format, label_format='xml', start_idx=0)
iterator = iter(generator.generate())
with K.get_session() as sess:

    for i in range(50):
        batch = next(iterator)
        img, label_true, _ = batch[0]
        _, label_pred, _ = batch[2]  # model.predict(img)

        print('True', label_true)
        print('Predicted', label_pred)

        label_true_t = np.expand_dims(model.encoder.encode_label(label_true), 0)
        label_pred_t = np.expand_dims(model.encoder.encode_label(label_pred), 0)
        y_true = K.placeholder(shape=[1, 13 * 13 * 5, 9], name='YTrue')
        y_pred = K.placeholder(shape=[1, 13 * 13 * 5, 9], name='YPred')
        confs = np.linspace(0.0, 1.0, 11)
        # K.tf.InteractiveSession()
        # K.tf.global_variables_initializer()
        dc = DetectionCountGateNet(batch_size=1, grid=[(3, 3)], confidence_levels=confs).compute(label_true_t,
                                                                                                 label_pred_t)
        label_true = model.decoder.decode_netout_to_label(label_true_t[0])
        label_pred = model.decoder.decode_netout_to_label(label_pred_t[0])
        label_true = ImgLabel([b for b in label_true.objects if b.confidence > 0])
        sess.run(K.tf.global_variables_initializer())

        tp, fp, fn = sess.run(dc, {y_true: label_true_t,
                                   y_pred: label_pred_t})
        for j, c in enumerate(confs):
            result_t = DetectionResult(true_positives=int(tp[0][j]), false_positives=int(fp[0][j]),
                                       false_negatives=int(fn[0][j]))
            label_pred = ImgLabel([b for b in label_pred.objects if b.confidence > c])
            print('Tensor:' + str(result_t))
            result = DetectionEvaluator().evaluate(label_true, label_pred)
            print('MetricDetection: ' + str(result))

            if result_t.n_fn != result.false_negatives or \
                    result_t.n_tp != result.true_positives or \
                    result_t.n_fp != result.false_positives:
                print('Mismatch!')
                t = 0
            else:
                t = 1

            show(img, labels=[label_pred, label_true], colors=[COLOR_BLUE, COLOR_GREEN],t=t)
