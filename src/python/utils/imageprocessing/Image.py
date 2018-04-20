import cv2


class Image:
    def __init__(self, array, color_format, path='./'):
        self.path = path
        self.format = color_format
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    @property
    def yuv(self):
        return self if self.format is 'yuv' else Image(cv2.cvtColor(self.array, cv2.COLOR_BGR2YUV), 'yuv')

    @property
    def bgr(self):
        return self if self.format is 'bgr' else Image(cv2.cvtColor(self.array, cv2.COLOR_YUV2BGR), 'bgr')

    def copy(self):
        return Image(self.array.copy(), self.format)
