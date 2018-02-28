from fileaccess.VocGenerator import VocGenerator

from fileaccess.GateGenerator import GateGenerator
from frontend.models.ssd.SSD import SSD
from frontend.models.yolo.Yolo import Yolo

from workdir import work_dir

work_dir()

from frontend.visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/mult_gate_aligned_test/', batch_size=1, color_format='bgr',
                          shuffle=False, start_idx=1)

generator = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                         "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=8, color_format='bgr')

model = SSD.ssd300(n_classes=20, weight_file='logs/ssd_ferrari/ssd300.h5', conf_thresh=0.01, color_format='bgr')
# model = Yolo.yolo_v2(class_names=['gate'], conf_thresh=0.05, color_format='yuv',
#                      weight_file='logs/yolov2_10k/YoloV2.h5')

demo_generator(model, generator, t_show=1)
