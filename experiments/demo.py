from models.Yolo.TinyYolo import TinyYolo

from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from models.Yolo.Yolo import Yolo
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/mult_gate_valid', batch_size=8, n_samples=100, color_format='bgr',
                          shuffle=True, start_idx=0)

model = TinyYolo(class_names=['gate'], weight_file='logs/tinyyolo-noaug/yolo-gate-adam.h5', conf_thresh=0.3)

demo_generator(model, generator, t_show=1)
