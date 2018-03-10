import random

from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import work_dir

work_dir()
# dataset = GateGenerator(directories=['resource/samples/cyberzoo_conv/'],
#                         batch_size=100, color_format='bgr',
#                         shuffle=False, start_idx=0, valid_frac=0,
#                         label_format=None).generate()
dataset = VocGenerator(batch_size=100).generate()
batch = next(dataset)

distortion_model = BarrelDistortion((600, 600), [.7, 0], use_newton_approx=True)

for img, label, _ in batch:
    img, label = resize(img, (600, 600), label=label)

    show(img, labels=label, t=1)
    img_d, label_d = distortion_model.distort(img, label)
    img_ud, label_ud = distortion_model.undistort(img_d, label_d)

    show(img_d, labels=label_d, name='distort', t=1)
    show(img_ud, labels=label_ud, name='restored')
