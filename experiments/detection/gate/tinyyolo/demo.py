from workdir import work_dir

work_dir()

from models.Yolo.TinyYolo import TinyYolo
from fileaccess.GateGenerator import GateGenerator
from visualization.utils import demo_generator

generator = GateGenerator(directory='resource/samples/mult_gate', batch_size=8, n_samples=100, color_format='yuv')

model = TinyYolo(class_names=['gate'], weight_file='logs/tiny-yolo-gate-mult-2/tiny-yolo-gate-adam.h5', conf_thresh=0.1)

demo_generator(model, generator)
