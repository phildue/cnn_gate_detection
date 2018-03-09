import glob
import os
import pickle
import random

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.LabelFileParser import LabelFileParser
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
                 label_format: str = 'pkl', n_samples=None, valid_frac=0.0, start_idx=0):
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

    def generate(self):
        return self.__generate(self.train_files)

    def generate_valid(self):
        return self.__generate(self.test_files)

    def __generate(self, files):
        current_batch = []
        files_it = iter(files)
        if not files:
            raise ValueError('GateGenerator::Cannot generate from empty list.')
        while True:
            try:
                file = next(files_it)
                img = imread(file, self.color_format)
                if self.label_format is not None:
                    label = LabelFileParser.read_file(self.label_format,
                                                      file.replace(self.img_format, self.label_format))
                else:
                    label = ImgLabel([])
                current_batch.append((img, label, file))
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            except StopIteration:
                if self.shuffle: random.shuffle(files)
                files_it = iter(files)

