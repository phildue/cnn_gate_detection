from fileaccess.ImgFileParser import ImgFileParser
from fileaccess.LabelFileParser import LabelFileParser
from labels.ImgLabel import ImgLabel

from src.python.utils.imageprocessing.Image import Image


def write_set(path: str, imgs: [Image], labels: [ImgLabel], img_format='jpg', label_format='pkl'):
    SetFileParser(path, img_format, label_format).write(imgs, labels)


def read_set(path: str, n=0, img_format='jpg', label_format='pkl') -> ([Image], [ImgLabel]):
    return SetFileParser(path, img_format, label_format).read(n)


class SetFileParser:
    def __init__(self, directory: str, img_format: str, label_format: str, start_idx=0):
        self.label_format = label_format
        self.img_format = img_format
        self.directory = directory
        self.sample_parser = ImgFileParser(self.directory, self.img_format, start_idx=start_idx)
        self.label_parser = LabelFileParser(self.directory, self.label_format, start_idx=start_idx)

    def read(self, n=0) -> ([Image], [ImgLabel]):
        return self.sample_parser.read(n), self.label_parser.read(n)

    def write(self, samples: [Image], labels: [ImgLabel]):
        self.sample_parser.write(samples)
        self.label_parser.write(labels)
