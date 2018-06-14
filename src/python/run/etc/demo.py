from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.visualization.demo import demo_generator
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.workdir import cd_work
import numpy as np

cd_work()

generator = CropGenerator(GateGenerator(directories=['resource/ext/samples/daylight'],
                                        batch_size=8, color_format='bgr',
                                        shuffle=False, start_idx=0, valid_frac=1.0,
                                        label_format='xml',
                                        ))
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
# model = Yolo.yolo_v2(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                      color_format='yuv', weight_file='logs/v2_mixed/model.h5')
# model = Yolo.tiny_yolo(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                        color_format='yuv', weight_file='logs/tiny_mixed/model.h5')
model = GateNet.create('GateNet3x3', weight_file='out/gate_crop3x3/model.h5', batch_size=8, norm=(52, 52), grid=(3, 3),
                       anchors=np.array([[1.68473955, 2.37206587],
                                         [3.20117521, 5.0159188],
                                         [0.99090763, 1.45475654],
                                         [4.90625, 7.18339932],
                                         [2.53021299, 3.40006656]]))

demo_generator(model, generator, t_show=0, n_samples=150, size=(52, 52))
