from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur

from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

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
