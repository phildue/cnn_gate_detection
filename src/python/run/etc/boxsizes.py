import numpy as np

from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.labels.utils import resize_label

bb = BoundingBox(1)
bb.w = 416
bb.h = 416

bb.cx = 208
bb.cy = 208
print(bb)
img = Image(np.zeros((416, 416, 3)), 'bgr')

box_sizes = [0.001, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
aspect_ratios = [4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.25]
for b in box_sizes:
    label = BoundingBox.to_label([bb])
    label = resize_label(label, scale_x=b, scale_y=b)
    box = BoundingBox.from_label(label)[0]
    print(box.area)
    print(box.area / (416 * 416))
    show(img, labels=label)

for a in aspect_ratios:
    label = BoundingBox.to_label([bb])
    label = resize_label(label, scale_x=0.1, scale_y=0.1)
    label = resize_label(label, scale_x=1.0, scale_y=a)
    box = BoundingBox.from_label(label)[0]
    print(box.h1 / box.w1)
    show(img, labels=label)
