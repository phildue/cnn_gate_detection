# https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformMotionBlur(ImgTransform):

    def __init__(self, direction, sigma=1, kernel_size=15):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.kernels = {
            'horizontal': TransformMotionBlur.kernel_horizontal,
            'vertical': TransformMotionBlur.kernel_vertical,
        }

        self.direction = direction

    def transform(self, img: Image, label: ImgLabel):
        img_t = img.copy()
        label_t = label.copy()

        if self.direction in self.kernels.keys():
            img_t = self._simple_transform(img_t)
        else:
            raise ValueError(self.direction)

        return img_t, label_t

    def _simple_transform(self, img_t: Image):
        kernel = self.kernels[self.direction](self.kernel_size, self.sigma)
        img_t.array = cv2.filter2D(img_t.array, -1, kernel)
        return img_t

    @staticmethod
    def kernel_horizontal(kernel_size, sigma):
        kernel_offset = int((kernel_size - 1) / 2)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_offset, :] = np.ones(kernel_size) * sigma
        kernel /= kernel_size
        return kernel

    @staticmethod
    def kernel_vertical(kernel_size, sigma):
        kernel_offset = int((kernel_size - 1) / 2)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, kernel_offset] = np.ones(kernel_size) * sigma
        kernel /= kernel_size
        return kernel
