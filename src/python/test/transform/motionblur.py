import random

from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.RandomColorNoise import RandomColorNoise
from utils.imageprocessing.transform.RandomMerge import RandomMerge
from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur
from utils.imageprocessing.transform.RandomRotate import RandomRotate
from utils.imageprocessing.transform.TransformDistort import TransformDistort

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.imageprocessing.transform.TransformerBlur import TransformerBlur
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomColorShift import RandomColorShift
from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransfromGray import TransformGray
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.TransformNormalize import TransformNormalize
from utils.imageprocessing.transform.TransformSubsample import TransformSubsample
from utils.imageprocessing.transform.RandomScale import RandomScale
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.SSDAugmenter import SSDAugmenter
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work
from utils.imageprocessing.Backend import resize, imread

cd_work()
sigma = 1.0
img_org = imread('resource/samples/norway.jpeg', 'bgr')
show(img_org, 'org')

img, _ = TransformMotionBlur('vertical', sigma).transform(img_org, ImgLabel([]))
show(img, 'vertical')

img, _ = TransformMotionBlur('horizontal', sigma).transform(img_org, ImgLabel([]))
show(img, 'horizontal')

img, _ = RandomMotionBlur().transform(img_org, ImgLabel([]))
show(img, 'random')
