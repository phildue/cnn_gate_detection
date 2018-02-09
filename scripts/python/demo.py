from fileaccess.VocGenerator import VocGenerator

from models.ssd.SSD import SSD
from models.yolo.TinyYolo import TinyYolo

from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from models.yolo.Yolo import Yolo
from visualization.utils import demo_generator

# generator = GateGenerator(directory='resource/samples/stream_valid1/', batch_size=8, color_format='bgr',
#                          shuffle=False, start_idx=400)


generator = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                         "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=8)

model = SSD.ssd300(n_classes=20, weight_file='logs/SSD300/SSD300.h5', conf_thresh=0.9)

demo_generator(model, generator, t_show=0)
