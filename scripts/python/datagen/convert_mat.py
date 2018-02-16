import glob

import numpy as np
from fileaccess.SetFileParser import write_set

from fileaccess.LabelFileParser import write_label

from imageprocessing.Image import Image

from labels.Pose import Pose

from imageprocessing.Backend import imread

from imageprocessing.Imageprocessing import show

from labels.ImgLabel import ImgLabel

from scene.GateCorners import GateCorners

from labels.GateLabel import GateLabel

from workdir import work_dir

work_dir()
import scipy.io

path = 'resource/samples/cyberzoo/'
mat = scipy.io.loadmat('resource/samples/cyberzoo/2018_2_2_ground_truth_gate_selection.mat')

labels = mat['GT_gate']
images = [imread(img, 'bgr') for img in list(sorted(glob.glob(path + '*.jpg')))]

images = [Image(np.rot90(img.array)) for img in images]
ymax = images[0].shape[0]
img_labels = []
for i in range(labels.shape[0]):
    gate_labels = []
    if labels[i, 0] > 0:
        x1, x2, x3, x4, y1, y2, y3, y4 = labels.astype(np.int16)[i, 1:]
        y1 = ymax - y1
        y2 = ymax - y2
        y3 = ymax - y3
        y4 = ymax - y4
        cx = int(x1 + (x3 - x1) / 2)
        cy = int(y1 + (y3 - y1) / 2)
        gate_label = GateLabel(position=Pose(),
                               gate_corners=GateCorners((cx, cy), (x4, y4), (x3, y3), (x2, y2), (x1, y1)))
        gate_labels.append(gate_label)

    img_labels.append(ImgLabel(gate_labels))

# for i in range(labels.shape[0]):
#     show(images[i], labels=img_labels[i])

# write_label(path, img_labels)
write_set('resource/samples/cyberzoo_conv/', images, img_labels)
