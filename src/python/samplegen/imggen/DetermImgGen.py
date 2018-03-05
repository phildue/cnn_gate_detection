import glob

from imageprocessing.Backend import COLOR_BGR2YUV
from imageprocessing.Backend import imread, replace_background, blur, resize, convert_color
from labels.ImgLabel import ImgLabel

from src.python.samplegen.imggen.ImgGen import ImgGen
from src.python.utils.imageprocessing.Image import Image


class DetermImgGen(ImgGen):
    def __init__(self, background_path="../resource/backgrounds", blur_kernel=(5, 5), output_shape=(416, 416),
                 convert_to_box_label=True, out_format='bgr'):
        self.out_format = out_format
        self.convert_to_box_label = convert_to_box_label
        self.output_shape = output_shape
        self.blur_kernel = blur_kernel
        paths = background_path if isinstance(background_path, list) else [background_path]
        self.files = list(sorted([f for folder in [glob.glob(p + "/*.jpg") for p in paths] for f in folder]))

    def generate(self, shots: [Image], labels: [ImgLabel], n_backgrounds=10) -> (
            [Image], [ImgLabel]):
        labels_created = []
        samples = []
        for j in range(len(shots)):
            for i in range(n_backgrounds):
                background = imread(self.files[i])
                img = replace_background(shots[j], background)
                img = blur(img, self.blur_kernel)

                if self.out_format is 'yuv':
                    img = convert_color(img, COLOR_BGR2YUV)

                img, label = resize(img, shape=self.output_shape, label=labels[j])
                samples.append(img)
                labels_created.append(label)

        return samples, labels_created
