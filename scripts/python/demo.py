from fileaccess.VocGenerator import VocGenerator

from fileaccess.GateGenerator import GateGenerator
from frontend.models.ssd.SSD import SSD
from frontend.models.yolo.Yolo import Yolo

from workdir import work_dir

work_dir()

from frontend.visualization.utils import demo_generator

generator = GateGenerator(directory=['resource/samples/cyberzoo_conv'],
                          batch_size=1, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0)
#
# generator = VocGenerator(batch_size=8, color_format='bgr')

# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc/SSD300.h5',iou_thresh_nms=0.3)
model = Yolo.tiny_yolo(class_names=['gate'], conf_thresh=0.6, color_format='yuv',
                       weight_file='logs/tinyyolo_aligned_distort_bs8/TinyYolo.h5')

demo_generator(model, generator, t_show=1)
