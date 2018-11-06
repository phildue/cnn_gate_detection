from pprint import pprint

from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.visuals.demo import demo_generator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
# 'resource/ext/samples/iros2018_course_final_simple_17gates'
generator = GateGenerator(directories=['resource/ext/samples/iros2018_course_final_simple_17gates'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=1.0,
                          label_format='xml',
                          img_format='jpg'
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
src_dir = 'out/thesis/datagen/yolov3_gate_realbg416x416_i00/'
summary = load_file(src_dir + 'summary.pkl')
pprint(summary['architecture'])
model = GateNet.create_by_arch(architecture=summary['architecture'],
                               weight_file=src_dir + 'model.h5', batch_size=8, norm=(416, 416),
                               anchors=summary['anchors'],
                               conf_thresh=0.3,
                               color_format='bgr',
                               # color_format='yuyv',
                               # preprocessor=TransformRaw(),
                               # input_channels=2,
                               augmenter=None)
# _model = build_detector((480, 640, 3), architecture=summary['architecture'], anchors=summary['anchors'])
# _model.load_weights(src_dir + '/model.h5')
# model.net.backend = _model
# create_dirs(['out/1807/narrow_strides_late_bottleneck416x416-13x13+9layers/img04/'])
demo_generator(model, generator, t_show=0, n_samples=2000, iou_thresh=0.6)
