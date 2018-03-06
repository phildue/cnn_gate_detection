from modelzoo.models.ssd.SSD import SSD
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.workdir import work_dir

work_dir()

from src.python.modelzoo.visualization.utils import demo_generator

generator = GateGenerator(directory=['resource/samples/mult_gate_aligned_blur/'],
                          batch_size=10, color_format='yuv',
                          shuffle=False, start_idx=0, valid_frac=0.2,
                          label_format='xml')
#
# generator = VocGenerator(batch_size=8, color_format='bgr')
#
model = SSD.ssd7(n_classes=1, conf_thresh=0.01, color_format='yuv', weight_file='logs/ssd7_gate/SSD7.h5',
                 iou_thresh_nms=0.3)
# model = Yolo.tiny_yolo(norm=(208, 208), grid=(6, 6), class_names=['gate'], conf_thresh=0.3, color_format='yuv',
#                        weight_file='logs/tinyyolo_aligned_distort_208/TinyYolo.h5')

demo_generator(model, generator, t_show=0)
