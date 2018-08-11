from pprint import pprint

from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.visualization.demo import demo_generator
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file, create_dirs
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work
import numpy as np

cd_work()

generator = GateGenerator(directories=['resource/ext/samples/industrial_new_test'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml',
                          )
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
# model = Yolo.yolo_v2(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                      color_format='yuv', weight_file='logs/v2_mixed/model.h5')
# model = Yolo.tiny_yolo(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                        color_format='yuv', weight_file='logs/tiny_mixed/model.h5')
src_dir = 'out/0108/corner416x416/'
summary = load_file(src_dir + 'summary.pkl')
pprint(summary['architecture'])
model = GateNet.create_by_arch(architecture=summary['architecture'],
                               weight_file=src_dir + 'model.h5', batch_size=8, norm=summary['img_res'],
                               anchors=summary['anchors'],
                               color_format='yuv',
                               # preprocessor=TransformGray(),
                               conf_thresh=0.6
                               )
# create_dirs(['out/1807/narrow_strides_late_bottleneck416x416-13x13+9layers/img04/'])
demo_generator(model, generator, t_show=0, n_samples=150, iou_thresh=0.4)
