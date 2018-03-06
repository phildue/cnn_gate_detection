from modelzoo.augmentation.Augmenter import Augmenter
from utils.imageprocessing.Backend import noisy, blur
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AugmenterBlur(Augmenter):
    def __init__(self, kernel=(5, 5), iterations=1):
        self.iterations = iterations
        self.kernel = kernel

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_aug = blur(img_aug, self.kernel, self.iterations)
        return img_aug
