from models.yolo.TinyYolo import TinyYolo

from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from models.yolo.Yolo import Yolo
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/stream_valid1/', batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=400)

model = TinyYolo(class_names=['gate'], weight_file='logs/tinyyolo-noaug/yolo-gate-adam.h5', conf_thresh=0.6)

demo_generator(model, generator, t_show=0)
