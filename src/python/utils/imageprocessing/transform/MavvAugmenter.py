from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomNoise import RandomNoise
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.imageprocessing.transform.TransformFlip import TransformFlip


class MavvAugmenter(RandomEnsemble):
    def __init__(self):
        augmenters = [(1.0, RandomBrightness(0.5, 2.0)),
                      (1.0, RandomNoise(0, 10)),
                      (1.0, RandomBlur((5, 5), 0, 10)),
                      (0.5, TransformFlip()),
                      (0.2, RandomShift(-.3, .3))]

        super().__init__(augmenters)
