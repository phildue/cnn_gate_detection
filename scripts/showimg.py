import os
import sys

from fileaccess.GateGenerator import GateGenerator
from imageprocessing.Imageprocessing import show

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.VocGenerator import VocSetParser

from imageprocessing.Backend import \
    annotate_bounding_box, resize, convert_color, COLOR_YUV2BGR
from shotgen.ShotLoad import ShotLoad


def show_voc():
    dataset = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                           "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=2).generate()
    for batch in dataset:
        for sample in batch:
            ann_img = annotate_bounding_box(sample[0], sample[1])
            show(ann_img, 'labeled')
            img, label = sample[0], sample[1]

            # img, label = resize(sample[0], shape=(500, 500), label=sample[1])
            # img, label = translate(img, shift_x=-10,shift_y= -10,label= label)
            # img, label = flip(img, label, 1)
            # ann_img = annotate_bounding_box(img, label)
            # show(ann_img, 'resized')


def show_img(path="resource/samples/stream_valid/"):
    gate_generator = GateGenerator(path, 8, color_format='bgr',shuffle=False)

    for batch in gate_generator.generate():
        for img, label in batch:
            if img.format == 'yuv':
                img = convert_color(img, COLOR_YUV2BGR)
            show(img, 'labeled', labels=label)


def show_shot(path="samplegen/resource/shots/stream/"):
    shots, labels = ShotLoad(shot_path=path, img_format='bmp').get_shots(
        n_shots=200)
    for i, img in enumerate(shots):
        ann_img, label = resize(img, (500, 500), label=labels[i])
        show(ann_img, 'labeled', labels=label)


#show_shot(path="samplegen/resource/shots/mult_gate_valid/")
show_img(path="resource/samples/mult_gate_valid/")
