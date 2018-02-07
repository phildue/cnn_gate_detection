from models.ssd.SSD import SSD
from models.yolo.TinyYolo import TinyYolo

from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from models.yolo.Yolo import Yolo
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/stream_valid1/', batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=400)

model = SSD(img_shape=(416, 416, 3), n_classes=1, weight_file='logs/ssd7/SSD7.h5')

demo_generator(model, generator, t_show=0)
