import random

from frontend.augmentation.AugmenterBrightness import AugmenterBrightness
from frontend.augmentation.AugmenterColorShift import AugmenterColorShift
from frontend.augmentation.AugmenterFlip import AugmenterFlip
from frontend.augmentation.AugmenterGray import AugmenterGray
from frontend.augmentation.AugmenterHistEq import AugmenterHistEq
from frontend.augmentation.AugmenterPixel import AugmenterPixel
from frontend.augmentation.AugmenterScale import AugmenterScale
from frontend.augmentation.AugmenterTranslate import AugmenterTranslate
from frontend.augmentation.SSDAugmenter import SSDAugmenter
from imageprocessing.Imageprocessing import show
from workdir import work_dir

from src.python.modelzoo.augmentation.AugmenterCrop import AugmenterCrop
from src.python.utils.fileaccess import VocGenerator

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
