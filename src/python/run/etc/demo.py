from modelzoo.models.ssd.SSD import SSD
from modelzoo.models.yolo.Yolo import Yolo
from modelzoo.visualization.demo import demo_generator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work
import numpy as np

cd_work()

generator = GateGenerator(directories=['resource/ext/samples/industrial_room_cats_test'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml',
                          )
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
model = Yolo.tiny_yolo(class_names=['gate'], batch_size=8, conf_thresh=0.1,
                     color_format='bgr', weight_file='logs/tiny_industrial_cats/TinyYolo.h5')
create_dirs(['logs/tiny_industrial_cast/demo/'])
demo_generator(model, generator, t_show=1,out_file='logs/tiny_industrial_cats/demo/')
