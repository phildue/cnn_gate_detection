from modelzoo.backend.tensor.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.backend.tensor.gatenet.DetectionCountYolo import DetectionCountGateNet
from modelzoo.evaluation import evaluate_generator, evaluate_file
from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work
import keras.backend as K
import numpy as np

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
                          shuffle=False, color_format=color_format, label_format='xml', start_idx=0)

batch = next(generator.generate())

i = 0
img, label_true, _ = batch[i]
label_pred = model.predict(img)

print(label_true)
label_true_t = np.expand_dims(model.encoder.encode_label(label_true), 0)
label_pred_t = np.expand_dims(model.encoder.encode_label(label_pred), 0)
y_true = K.placeholder(shape=[1, 13 * 13 * 5, 9], name='YTrue')
y_pred = K.placeholder(shape=[1, 13 * 13 * 5, 9], name='YPred')
dc = DetectionCountGateNet(batch_size=1, grid=[(3, 3)], confidence_levels=[0.3]).compute(y_true,
                                                                                         y_pred)
label_true = model.decoder.decode_netout_to_label(label_true_t)
print(label_true)
label_pred = model.decoder.decode_netout_to_label(label_pred_t)
with K.get_session() as sess:
    sess.run(K.tf.global_variables_initializer())

    tp, fp, fn = sess.run(dc, {y_true: label_true_t,
                               y_pred: label_pred_t})
    result_t = DetectionResult(true_positives=int(tp[i][0]), false_positives=int(fp[i][0]),
                               false_negatives=int(fn[i][0]))
    print('Tensor:' + str(result_t))
    result = MetricDetection().evaluate(label_true, label_pred)
    print('MetricDetection: ' + str(result))