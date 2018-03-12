import random



from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformerBlur import TransformerBlur
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomColorShift import RandomColorShift
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.RandomNoise import RandomNoise
from utils.imageprocessing.transform.TransformNormalize import TransformNormalize
from utils.imageprocessing.transform.TransformSubsample import TransformSubsample
from utils.imageprocessing.transform.RandomScale import RandomScale
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.SSDAugmenter import SSDAugmenter
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
augmenters = [RandomCrop(), RandomBrightness(), RandomColorShift(), TransformFlip(), TransformGray(),
              TransformHistEq(), TransformSubsample(), RandomNoise(iterations=10), TransformerBlur(iterations=10),
              RandomScale(), RandomShift(), TransformNormalize()]

for img, label, _ in batch:
    show(img, labels=label, name='Org')
    for augmenter in augmenters:
        img_aug, label_aug = augmenter.transform(img, label)
        show(img_aug, name=augmenter.__class__.__name__)
