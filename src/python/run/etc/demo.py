from modelzoo.models.ssd.SSD import SSD
from modelzoo.models.yolo.Yolo import Yolo
from modelzoo.visualization.demo import demo_generator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.workdir import cd_work
import numpy as np

cd_work()

generator = GateGenerator(directories=['resource/samples/cyberzoo/'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml')
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
anchors = np.array([[0.13809687, 0.27954467],
                    [0.17897748, 0.56287585],
                    [0.36642611, 0.39084195],
                    [0.60043528, 0.67687858]])

model = Yolo.yolo_v2(norm=(160, 320), grid=(5, 10), class_names=['gate'], batch_size=8, conf_thresh=0.1,
                     color_format='bgr', anchors=anchors, weight_file='logs/v2_bebop/YoloV2.h5')
demo_generator(model, generator, t_show=0)
