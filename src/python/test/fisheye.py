import random

from imageprocessing.Backend import resize
from imageprocessing.Imageprocessing import show
from imageprocessing.fisheye import fisheye
from workdir import work_dir

from src.python.utils.fileaccess import VocGenerator

work_dir()

dataset = VocGenerator(batch_size=100).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, 99 - N)

img = batch[idx][0]
label = batch[idx][1]
img, label = resize(img, (300, 300), label=label)
show(img, labels=label)

img_fisheye, label_eye = fisheye(img, k=0.000001 * 30, label_eye=label)

show(img_fisheye, labels=label_eye)
