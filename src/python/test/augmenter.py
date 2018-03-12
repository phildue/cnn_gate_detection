import random



from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.augmentation.AugmenterBlur import AugmenterBlur
from utils.imageprocessing.augmentation.AugmenterBrightness import AugmenterBrightness
from utils.imageprocessing.augmentation.AugmenterColorShift import AugmenterColorShift
from utils.imageprocessing.augmentation.AugmenterCrop import AugmenterCrop
from utils.imageprocessing.augmentation.AugmenterFlip import AugmenterFlip
from utils.imageprocessing.augmentation.AugmenterGray import AugmenterGray
from utils.imageprocessing.augmentation.AugmenterHistEq import AugmenterHistEq
from utils.imageprocessing.augmentation.AugmenterNoise import AugmenterNoise
from utils.imageprocessing.augmentation.AugmenterNormalize import AugmenterNormalize
from utils.imageprocessing.augmentation.AugmenterPixel import AugmenterPixel
from utils.imageprocessing.augmentation.AugmenterScale import AugmenterScale
from utils.imageprocessing.augmentation.AugmenterTranslate import AugmenterTranslate
from utils.imageprocessing.augmentation.SSDAugmenter import SSDAugmenter
from utils.workdir import work_dir

work_dir()

generator = GateGenerator(directories=['resource/samples/cyberzoo/'],
                          batch_size=100, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0,
                          label_format='xml')
batch = next(generator.generate())
idx = random.randint(0, 80)
img = batch[idx][0]
label = batch[idx][1]

ssd_augmenter = SSDAugmenter()
augmenters = [AugmenterCrop(), AugmenterBrightness(), AugmenterColorShift(), AugmenterFlip(), AugmenterGray(),
              AugmenterHistEq(), AugmenterPixel(), AugmenterNoise(iterations=10), AugmenterBlur(iterations=10),
              AugmenterScale(), AugmenterTranslate(), AugmenterNormalize()]

for img, label, _ in batch:
    show(img, labels=label, name='Org')
    for augmenter in augmenters:
        img_aug, label_aug = augmenter.augment(img, label)
        show(img_aug, name=augmenter.__class__.__name__)
