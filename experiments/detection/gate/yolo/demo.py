from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from models.Yolo.Yolo import Yolo
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/iros16', batch_size=8, n_samples=100, color_format='bgr',
                          shuffle=False, start_idx=2000)

model = Yolo(class_names=['gate'], weight_file='logs/yolo/yolo-gate-adam.h5', conf_thresh=0.3)

demo_generator(model, generator, t_show=1)
