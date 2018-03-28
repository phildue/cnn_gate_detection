import glob

import numpy as np
import scipy.io

from utils.fileaccess.SetFileParser import write_set
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Image import Image
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.Pose import Pose
from utils.workdir import cd_work

cd_work()

path = '../gate_detection/basement/'
out_path = 'resource/samples/basement/'
mat = scipy.io.loadmat('../gate_detection/3_19_GT_gate.mat')

labels = mat['GT_gate']
images = [imread(img, 'bgr') for img in list(sorted(glob.glob(path + '*.jpg')))]

images = [Image(np.rot90(img.array), 'bgr') for img in images]
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
create_dirs([out_path])
write_set(out_path, images, img_labels, label_format='xml')
