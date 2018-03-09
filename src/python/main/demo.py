from modelzoo.models.ssd.SSD import SSD
from modelzoo.visualization.utils import demo_generator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.workdir import work_dir

work_dir()

generator = GateGenerator(directories=['resource/samples/video/eth/'],
                          batch_size=10, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml')
#
generator = VocGenerator(batch_size=8)

model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
                   iou_thresh_nms=0.3)
# model = Yolo.yolo_v2(class_names=['gate'], conf_thresh=0.2, color_format='yuv',
#                      weight_file='logs/yolov2_20k_new/YoloV2.h5')

demo_generator(model, generator, t_show=0)
