import random

from modelzoo.augmentation.AugmenterBrightness import AugmenterBrightness
from modelzoo.augmentation.AugmenterColorShift import AugmenterColorShift
from modelzoo.augmentation.AugmenterFlip import AugmenterFlip
from modelzoo.augmentation.AugmenterGray import AugmenterGray
from modelzoo.augmentation.AugmenterPixel import AugmenterPixel

from modelzoo.augmentation.AugmenterCrop import AugmenterCrop
from modelzoo.augmentation.AugmenterHistEq import AugmenterHistEq
from modelzoo.augmentation.AugmenterScale import AugmenterScale
from modelzoo.augmentation.AugmenterTranslate import AugmenterTranslate
from modelzoo.augmentation.SSDAugmenter import SSDAugmenter

from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import work_dir

work_dir()

dataset = VocGenerator(batch_size=100).generate()
batch = next(dataset)
idx = random.randint(0, 80)
img = batch[idx][0]
label = batch[idx][1]

ssd_augmenter = SSDAugmenter()
augmenters = [AugmenterCrop(), AugmenterBrightness(), AugmenterColorShift(), AugmenterFlip(), AugmenterGray(),
              AugmenterHistEq(), AugmenterPixel(),
              AugmenterScale(), AugmenterTranslate()]

for img, label, _ in batch:
    show(img, labels=label, name='Org')
    for augmenter in augmenters:
        img_aug, label_aug = augmenter.augment(img, label)
        show(img_aug, labels=label_aug, name=augmenter.__class__.__name__)
