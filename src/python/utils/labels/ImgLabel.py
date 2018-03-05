from src.python.utils.labels import ObjectLabel


class ImgLabel:
    def __init__(self, objects: [ObjectLabel]):
        self.objects = objects
