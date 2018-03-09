import random

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.BarrelDistortion import BarrelDistortion

from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize, imread
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.fisheye import fisheye
from utils.workdir import work_dir
import numpy as np

work_dir()
# dataset = GateGenerator(directories=['resource/samples/cyberzoo_conv/'],
#                         batch_size=100, color_format='bgr',
#                         shuffle=False, start_idx=0, valid_frac=0,
#                         label_format=None).generate()
dataset = VocGenerator(batch_size=100).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, 99 - N)

img = batch[idx][0]
label = batch[idx][1]
img, label = resize(img, (600, 600), label=label)

img = imread('resource/samples/fisheye.jpg', 'bgr')
test_mat = np.zeros((300, 300, 3))
for i in range(0, 300, 10):
    test_mat[i, :] = (255, 255, 255)
for j in range(0, 300, 10):
    test_mat[:, j] = (255, 255, 255)
test_mat = 255 - test_mat
img = Image(test_mat, 'bgr')
distortion_model = BarrelDistortion(img.shape[:2], [.2, 0], max_iterations=100, distortion_radius=1.5)
show(img, labels=label, t=1)

img_ud, label_ud = distortion_model.undistort(img, label)
img_d, label_d = distortion_model.distort(img, label)
img_rd, label_rd = distortion_model.undistort(img_d, label)

show(img_ud, labels=label_ud, name='undistort', t=1)

show(img_d, labels=label_d, name='distort', t=1)
show(img_rd, labels=label_rd, name='restored')
