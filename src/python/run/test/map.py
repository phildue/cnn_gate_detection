from modelzoo.backend.tensor.gatenet.DetectionCountYolo import DetectionCountGateNet
from modelzoo.evaluation import evaluate_generator, evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work
import keras.backend as K

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(500 / batch_size)
image_source = ['resource/ext/samples/industrial_new_test/']
color_format = 'bgr'

# Model
conf_thresh = 0

model = ModelFactory.build('GateNet3x3', batch_size=batch_size, img_res=(52, 52), grid=[(3, 3)])

# Evaluator
iou_thresh = 0.4

# Result Paths
generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=False, color_format=color_format, label_format='xml', start_idx=0)

batch = next(generator.generate())

for i in range(batch_size):
    img, label_true, _ = batch[i]
    label_pred = model.predict(img)
    result = MetricDetection().evaluate(label_true, label_pred)
    print('MetricDetection: ' + str(result))

    label_true_t = model.encoder.encode_label(label_true)
    label_pred_t = model.encoder.encode_label(label_pred)

    result_t = K.get_session().run(
        DetectionCountGateNet(batch_size=1, grid=[(3, 3)]).compute(K.expand_dims(K.constant(label_true_t), 0),
                                                                   K.expand_dims(K.constant(label_pred_t), 0)))
    print('Tensor Metric:' + result_t)
