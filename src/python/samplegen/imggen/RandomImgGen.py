import glob
import random

from samplegen.imggen.ImgGen import ImgGen
from utils.imageprocessing.Backend import imread, replace_background, blur, noisy, convert_color, COLOR_BGR2YUV, resize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class RandomImgGen(ImgGen):
    def __init__(self, background_path="../resource/backgrounds", blur_kernel=(5, 5), blur_it=10,
                 output_shape=(416, 416),
                 out_format='bgr', noisy_var=30.0, noisy_it=10):
        self.noisy_var = noisy_var
        self.noisy_it = noisy_it
        self.blur_it = blur_it
        self.out_format = out_format
        self.output_shape = output_shape
        self.blur_kernel = blur_kernel
        paths = background_path if isinstance(background_path, list) else [background_path]
        self.files = [f for folder in [glob.glob(p + "/*.jpg") for p in paths] for f in folder]

    def generate(self, shots: [Image], labels: [ImgLabel], n_backgrounds=10) -> (
            [Image], [ImgLabel]):
        labels_created = []
        samples = []
        for j in range(n_backgrounds):
            background = imread(random.choice(self.files))
            i = random.randint(0, len(shots) - 1)
            shot = shots[i]

            img = replace_background(shot, background)
            img = blur(img, self.blur_kernel, self.blur_it)
            img = noisy(img, iterations=self.noisy_it, var=self.noisy_var)

            if self.out_format is 'yuv':
                img = convert_color(img, COLOR_BGR2YUV)

            img, label = resize(img, shape=self.output_shape, label=labels[i])
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
