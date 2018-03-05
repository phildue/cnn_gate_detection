from frontend.augmentation.AugmenterBrightness import AugmenterBrightness
from frontend.augmentation.AugmenterEnsemble import AugmenterEnsemble
from frontend.augmentation.AugmenterFlip import AugmenterFlip

from src.python.modelzoo.augmentation.AugmenterCrop import AugmenterCrop


class SSDAugmenter(AugmenterEnsemble):
    def __init__(self, p_equalize=0.5, p_flip=0.5,
                 p_translate=0.5, p_scale=0.5, p_gray=0.5, p_brightness=0.5, p_crop=0.5):
        augmenters = [(p_flip, AugmenterFlip()),
                      (p_brightness, AugmenterBrightness(0.5, 2.0)),
                      (p_crop, AugmenterCrop(c_max=0.9, c_min=0.7))]

        super().__init__(augmenters)
