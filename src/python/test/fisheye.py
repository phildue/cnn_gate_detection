import random

from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.fisheye import fisheye
from utils.workdir import work_dir

work_dir()

dataset = VocGenerator(batch_size=100).generate()
batch = next(dataset)
N = 5
idx = random.randint(0, 99 - N)

img = batch[idx][0]
label = batch[idx][1]
img, label = resize(img, (300, 300), label=label)
show(img, labels=label)

img_fisheye, label_eye = fisheye(img, k=0.000001 * 30, label=label)

show(img_fisheye, labels=label_eye)
