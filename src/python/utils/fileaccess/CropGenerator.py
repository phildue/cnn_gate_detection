from random import randint, random

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.imageprocessing.Backend import crop


class CropGenerator(DatasetGenerator):

    def __init__(self, gate_generator: DatasetGenerator, top_crops=1):
        self.top_crops = top_crops
        self.gate_generator = gate_generator

    def generate(self):
        it = iter(self.gate_generator.generate())
        return self._generate(it)

    def _generate(self, iterator):
        while True:
            batch = next(iterator)
            batch_filtered = []
            for img, label, file in batch:
                areas = np.array([obj.area for obj in label.objects])
                filtered_objs = []
                if len(areas) > 0:
                    for i in range(self.top_crops):
                        filtered_objs.append(label.objects[np.argmax(areas)])

                for obj in filtered_objs:

                    crop_min = (max(0, obj.x_min - randint(10, 30)), max(obj.y_min - randint(10, 30), 0))
                    crop_max = (
                        min(obj.x_max + randint(10, 30), img.shape[1]), min(obj.y_max + randint(10, 30), img.shape[0]))

                    w = crop_max[0] - crop_min[0]
                    h = crop_max[1] - crop_min[1]
                    dw = 0
                    dh = 0
                    if w > h:
                        dw = h - w
                    elif h > w:
                        dh = w - h

                    crop_max = crop_max[0] + dw / 2, crop_max[1] + dw / 2,
                    crop_min = crop_min[0] - dh / 2, crop_min[1] - dh / 2

                    img_crop, label_crop = crop(img, crop_min, crop_max, label)
                    if img_crop.array.size > 0:
                        batch_filtered.append((img_crop, label_crop, file))

                if len(batch_filtered) >= self.batch_size:
                    yield batch_filtered
                    del batch_filtered
                    batch_filtered = []

    def generate_valid(self):
        it = iter(self.gate_generator.generate_valid())
        return self._generate(it)

    @property
    def n_samples(self):
        return self.gate_generator.n_samples

    @property
    def batch_size(self):
        return self.gate_generator.batch_size

    @property
    def source_dir(self):
        return self.gate_generator.source_dir

    @property
    def color_format(self):
        return self.gate_generator.color_format
