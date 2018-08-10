from utils.Polygon import Quadrangle
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Image import Image
import numpy as np

from utils.imageprocessing.Imageprocessing import show, COLOR_RED, COLOR_GREEN, COLOR_BLUE, LEGEND_CORNERS
from utils.workdir import cd_work

cd_work()
train_gen = GateGenerator(['resource/ext/samples/industrial_new_test/'], batch_size=10, valid_frac=0.05,
                          color_format='bgr', label_format='xml', n_samples=10)

batch = next(train_gen.generate())

label = batch[0][1]
q1 = Quadrangle.from_label(label)[0]

q2 = Quadrangle()
q2.wBottom = 100
q2.wTop = 100
q2.hLeft = 200
q2.hRight = 150

q2.cx = 300
q2.cy = 240
q2.class_conf = 1.0

print(q1.iou(q2))

black_image = Image(np.zeros((680, 680, 3)), 'bgr')

l1 = Quadrangle.to_label([q1])
l2 = Quadrangle.to_label([q2])

show(black_image, colors=[COLOR_BLUE, COLOR_RED], labels=[l1, l2], thickness=1)
