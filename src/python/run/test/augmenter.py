import random

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.RandomColorShift import RandomColorShift
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.RandomHSV import RandomHSV
from utils.imageprocessing.transform.RandomScale import RandomScale
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.SSDAugmenter import SSDAugmenter
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransformNormalize import TransformNormalize
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
              # TransformDistort(BarrelDistortion.from_file('resource/distortion_model_est.pkl')),
              TransformFlip(), TransformGray(),
              TransformHistEq(), TransformSubsample(), RandomGrayNoise(), TransformerBlur(iterations=10),
              RandomScale(), RandomShift(), TransformNormalize(), RandomCrop()]

augmenters = [RandomHSV((0.5, 2.0), (0.5, 2.0), (0.5, 2.0)), TransformFlip(), TransformGray(),
              TransformHistEq(), RandomGrayNoise(), TransformerBlur(iterations=10),
              TransformNormalize()]
augmenters = []

for img, label, _ in batch:
    img, label = resize(img, (180, 315), label=label)
    show(img, name='Org')

    for augmenter in augmenters:
        img_aug, label_aug = augmenter.transform(img, label)
        show(img_aug, name=augmenter.__class__.__name__, labels=label_aug)
