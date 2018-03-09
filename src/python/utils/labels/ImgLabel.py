from utils.labels.ObjectLabel import ObjectLabel
import copy


class ImgLabel:
    def __init__(self, objects: [ObjectLabel]):
        self.objects = objects

    def copy(self):
        return copy.deepcopy(self)
