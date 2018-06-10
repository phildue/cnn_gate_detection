from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.labels.ImgLabel import ImgLabel


class YoloParser(AbstractDatasetParser):
    def read(self, n=0) -> ([Image], [ImgLabel]):
        raise NotImplementedError()

    @staticmethod
    def read_label(filepath: str) -> [ImgLabel]:
        raise NotImplementedError()

    def write_label(self, label: ImgLabel, path: str):
        # [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
        with open(path + '.txt', "w+") as f:
            for obj in label.objects:
                f.write('{} {} {} {} {}\n'.format(obj.class_id, obj.cx, obj.cy, obj.width, obj.height))
