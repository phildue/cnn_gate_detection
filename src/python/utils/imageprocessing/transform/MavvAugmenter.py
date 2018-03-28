from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomMerge import RandomMerge
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.imageprocessing.transform.TransformFlip import TransformFlip


class MavvAugmenter(RandomEnsemble):
    def __init__(self):
        augmenters = [(1.0, RandomBrightness(0.5, 2.0)),
                      (1.0, RandomGrayNoise(0, 10)),
                      (1.0, RandomBlur((5, 5), 0, 10)),
                      (0.5, TransformFlip()),
                      (0.2, RandomShift(-.3, .3)),
                      (0.5, RandomMerge(pixel_frac=0.005, kernel_size=(9, 9)))]

        super().__init__(augmenters)
