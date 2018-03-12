import glob
import random

from samplegen.imggen.ImgGen import ImgGen
from utils.imageprocessing.Backend import imread, replace_background, blur, noisy, convert_color, COLOR_BGR2YUV, resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.augmentation.Augmenter import Augmenter
from utils.labels.ImgLabel import ImgLabel


class RandomImgGen(ImgGen):
    def __init__(self, background_path="../resource/backgrounds",
                 output_shape=(416, 416),
                 image_transformer: Augmenter = None):
        self.output_shape = output_shape
        paths = background_path if isinstance(background_path, list) else [background_path]
        self.files = [f for folder in [glob.glob(p + "/*.jpg") for p in paths] for f in folder]
        self.image_transformer = image_transformer

    def generate(self, shots: [Image], labels: [ImgLabel], n_backgrounds=10) -> (
            [Image], [ImgLabel]):
        labels_created = []
        samples = []
        for j in range(n_backgrounds):
            background = imread(random.choice(self.files), 'bgr')
            i = random.randint(0, len(shots) - 1)
            shot = shots[i]

            img = replace_background(shot, background)
            label = labels[i]

            if self.image_transformer:
                img, label = self.image_transformer.augment(img, label)

            img, label = resize(img, shape=self.output_shape, label=label)
            samples.append(img)
            labels_created.append(label)

        return samples, labels_created

    def gen_empty_samples(self, n_samples) -> ([Image], [ImgLabel]):
        labels_created = []
        samples = []
        for j in range(n_samples):
            background = imread(random.choice(self.files), 'bgr')

            samples.append(background)
            labels_created.append(ImgLabel([]))

        return samples, labels_created
