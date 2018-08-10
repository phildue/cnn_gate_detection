import glob
import os
import random

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.imageprocessing.Backend import imread
from utils.labels.ImgLabel import ImgLabel


class GateGenerator(DatasetGenerator):
    @property
    def color_format(self):
        return self._color_format

    @property
    def source_dir(self):
        return self.directories

    @property
    def batch_size(self):
        return self.__batch_size

    def __init__(self, directories: [str], batch_size: int, shuffle: bool = True, img_format: str = 'jpg',
                 color_format='yuv',
                 label_format: str = 'pkl', n_samples=None, valid_frac=0.0, start_idx=0, org_aspect_ratio=1.05,
                 max_angle=30, max_distance=20):
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.org_aspect_ratio = org_aspect_ratio
        self._color_format = color_format
        self.label_format = label_format
        self.__batch_size = batch_size
        self.shuffle = shuffle
        self.img_format = img_format
        self.directories = directories
        files_all = []
        for d in directories:
            files_dir = sorted(glob.glob(d + "/*." + img_format))
            files_dir = [os.path.abspath(f) for f in files_dir]
            files_all.extend(files_dir)

        files = files_all[start_idx:]
        if n_samples is not None:
            files = files[:n_samples]

        print('Gate Generator::{0:d} samples found. Using {1:d}'.format(len(files_all), len(files)))
        if shuffle:
            random.shuffle(files)

        self.train_files = files[:int(np.ceil((1 - valid_frac) * len(files)))]
        self.test_files = files[:int(np.floor(valid_frac * len(files)))]
        self.__n_samples = len(self.train_files)

    @property
    def n_samples(self):
        return self.__n_samples

    def __len__(self):
        return self.n_samples

    def generate(self):
        return self._generate(self.train_files)

    def generate_valid(self):
        return self._generate(self.test_files)

    def _generate(self, files):
        current_batch = []
        files_it = iter(files)
        if not files:
            raise ValueError('GateGenerator::Cannot generate from empty list.')
        while True:
            try:
                file = next(files_it)
                img = imread(file, self.color_format)
                if self.label_format is not None:
                    label = DatasetParser.read_label(self.label_format,
                                                     file.replace(self.img_format, self.label_format))
                    label = self._filter(label)

                else:
                    label = ImgLabel([])
                current_batch.append((img, label, file))
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    del current_batch
                    current_batch = []
            except StopIteration:
                if self.shuffle: random.shuffle(files)
                files_it = iter(files)

    def _filter(self, label: ImgLabel):
        max_aspect_ratio = self.org_aspect_ratio / (self.max_angle / 90)
        objs_filtered = [obj for obj in label.objects if obj.pose.north < self.max_distance and \
                         obj.height / obj.width < max_aspect_ratio]
        return ImgLabel(objs_filtered)
