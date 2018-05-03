from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.ssd.SSD import SSD
from modelzoo.models.yolo.Yolo import Yolo
from modelzoo.visualization.demo import demo_generator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work
import numpy as np

cd_work()

generator = GateGenerator(directories=['resource/ext/samples/daylight_flight'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml',
                          )
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
model = Yolo.yolo_v2(class_names=['gate'], batch_size=8, conf_thresh=0.5,
                     color_format='yuv', weight_file='logs/v2_mixed/model.h5')
# model = Yolo.tiny_yolo(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                        color_format='yuv', weight_file='logs/tiny_mixed/model.h5')

# model = GateNet.v8(weight_file='logs/gatev8_mixed/model.h5', conf_thresh=0.6, color_format='yuv')
create_dirs(['logs/gatev6_mixed/demo/daylight_test'])
demo_generator(model, generator, t_show=0, n_samples=150)
