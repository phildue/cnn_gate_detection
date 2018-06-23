from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.imageprocessing.Backend import resize
from utils.labels.ImgLabel import ImgLabel


class YoloParser(AbstractDatasetParser):

    def __init__(self, directory: str, color_format, image_format='jpg', start_idx=0, img_norm=(416, 416)):
        super().__init__(directory, color_format, start_idx, image_format)
        self.img_norm = img_norm

    def read(self, n=0) -> ([Image], [ImgLabel]):
        raise NotImplementedError()

    @staticmethod
    def read_label(filepath: str) -> [ImgLabel]:
        raise NotImplementedError()

    def write_label(self, label: ImgLabel, path: str):
        # [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
        with open(path + '.txt', "w+") as f:
            for obj in label.objects:
                f.write('{} {} {} {} {}\n'.format(obj.class_id, obj.cx / self.img_norm[1], obj.cy / self.img_norm[0],
                                                  obj.width / self.img_norm[1], obj.height / self.img_norm[0]))

    def write(self, images: [Image], labels: [ImgLabel]):
        for i, l in enumerate(labels):
            filename = '{}/{:05d}'.format(self.directory, self.idx)
            img, label = resize(images[i], self.img_norm, label=l)
            for obj in label.objects:
                obj.y_min = self.img_norm[0] - obj.y_min
                obj.y_max = self.img_norm[0] - obj.y_max
            if self.color_format is 'yuv':
                self.write_img(img.yuv, filename + '.' + self.image_format)
            elif self.color_format is 'bgr':
                self.write_img(img.bgr, filename + '.' + self.image_format)
            else:
                raise ValueError('Unknown color format')
            self.write_label(label, filename)
            self.idx += 1
