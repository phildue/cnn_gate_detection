import random

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import convert_color, COLOR_YUV2BGR, COLOR_YUYV2YUV
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomChromatic import RandomChromatic
from utils.imageprocessing.transform.RandomColorShift import RandomColorShift
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.RandomExposure import RandomExposure
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.RandomHSV import RandomHSV
from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur
from utils.imageprocessing.transform.RandomScale import RandomScale
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.SSDAugmenter import SSDAugmenter
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransformNormalize import TransformNormalize
from utils.imageprocessing.transform.TransformRaw import TransformRaw
from utils.imageprocessing.transform.TransformSubsample import TransformSubsample
from utils.imageprocessing.transform.TransformerBlur import TransformerBlur
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.workdir import cd_work

cd_work()

generator = GateGenerator(directories=['resource/ext/samples/daylight_course1/'],
                          batch_size=100, color_format='bgr',
                          shuffle=True, start_idx=0, valid_frac=0,
                          label_format='xml', img_format='jpg')
batch = next(generator.generate())
idx = random.randint(0, 80)
img = batch[idx][0]
label = batch[idx][1]
ssd_augmenter = SSDAugmenter()
augmenters = [RandomColorShift((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),  # RandomMerge(), RandomRotate(10, 30),
              # ,
              TransformFlip(), TransformGray(),
              TransformHistEq(), TransformSubsample(), RandomGrayNoise(), TransformerBlur(iterations=10),
              RandomScale(), RandomShift(), TransformNormalize(), RandomCrop()]

augmenters = [RandomHSV((0.5, 2.0), (0.5, 2.0), (0.5, 2.0)), TransformFlip(), TransformGray(),
              TransformHistEq(), RandomGrayNoise(), TransformerBlur(iterations=10),
              TransformNormalize()]
augmenters = [RandomExposure((0.5, 1.5)), RandomMotionBlur(1.0, 2.0), RandomBlur(),
              RandomHSV((.9, 1.1), (0.5, 1.5), (0.5, 1.5)), RandomChromatic((-2, 2), (0.99, 1.01), (-2, 2))]

augmenters = [
    TransformDistort(BarrelDistortion((416, 416), rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0]),
                     2.0),
    RandomExposure((0.8, 1.2)),
    RandomMotionBlur(1.0, 2.0, 15),
    RandomBlur((5, 5)),
    RandomHSV((.9, 1.1), (0.8, 1.2), (0.8, 1.2)),
    RandomChromatic((-2, 2), (0.99, 1.01), (-2, 2)),
    TransformRaw()
]
for img, label, _ in batch:
    # img, label = resize(img, (180, 315), label=label)
    show(img, name='Org')

    for augmenter in augmenters:
        img_aug, label_aug = augmenter.transform(img, label)
        if img_aug.format is 'yuyv':
            img_aug = convert_color(img_aug, COLOR_YUYV2YUV)
            img_aug = convert_color(img_aug, COLOR_YUV2BGR)
        if img_aug.format is 'bgr':
            show(img_aug, name=augmenter.__class__.__name__, labels=label_aug)
        else:
            print("Unknown format")
