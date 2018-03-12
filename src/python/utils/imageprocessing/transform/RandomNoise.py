from utils.imageprocessing.Backend import noisy
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomNoise(ImgTransform):
    def __init__(self, variance=10, iterations=1):
        self.iterations = iterations
        self.variance = variance

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_aug = noisy(img_aug, self.variance, self.iterations)
        return img_aug, label
