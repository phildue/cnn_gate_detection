import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class VocGenerator(DatasetGenerator):
    @property
    def color_format(self):
        return self._color_format

    @property
    def source_dir(self):
        return self.directory

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def n_samples(self):
        return self.__n_samples

    def __init__(self, dir_voc2012: str = "resource/ext/backgrounds/VOCdevkit/VOC2012/",
                 dir_voc2007='resource/ext/backgrounds/VOCdevkit/VOC2007/',
                 dir_voc2007_test='resource/ext/backgrounds/VOCdevkit/VOC2007_Test/',
                 batch_size: int = 10, shuffle: bool = True, n_samples=None, start_idx=0):

        self._color_format = 'bgr'
        self.shuffle = shuffle
        self.directory = [dir_voc2012, dir_voc2007]
        self.__batch_size = batch_size
        imgs_2007 = dir_voc2007 + "JPEGImages/"
        imgs_2007_test = dir_voc2007_test + "JPEGImages/"
        imgs_2012 = dir_voc2012 + "JPEGImages/"

        labels_2007 = dir_voc2007 + "Annotations/"
        labels_2007_test = dir_voc2007_test + "Annotations/"
        labels_2012 = dir_voc2012 + "Annotations/"

        set_2007_train = load_file(dir_voc2007 + "ImageSets/Main/train.txt").split('\n')
        set_2007_train = [(name, labels_2007, imgs_2007) for name in set_2007_train if len(name) > 2]

        set_2007_valid = load_file(dir_voc2007 + "ImageSets/Main/val.txt").split('\n')
        set_2007_valid = [(name, labels_2007, imgs_2007) for name in set_2007_valid if len(name) > 2]

        set_2012_train = load_file(dir_voc2012 + "ImageSets/Main/train.txt").split('\n')
        set_2012_train = [(name, labels_2012, imgs_2012) for name in set_2012_train if len(name) > 2]

        set_2012_valid = load_file(dir_voc2012 + "ImageSets/Main/val.txt").split('\n')
        set_2012_valid = [(name, labels_2012, imgs_2012) for name in set_2012_valid if len(name) > 2]

        set_2007_test = load_file(dir_voc2007_test + "ImageSets/Main/test.txt").split('\n')
        set_2007_test = [(name, labels_2007_test, imgs_2007_test) for name in set_2007_test if len(name) > 2]

        self.files = set_2007_train + set_2007_valid + set_2012_train + set_2007_test
        self.test_files = set_2012_valid

        if n_samples is not None:
            self.files = self.files[start_idx:n_samples]
        else:
            self.files = self.files[start_idx:]

        if shuffle:
            random.shuffle(self.files)
            random.shuffle(self.test_files)

        print('VOCGenerator::{0:d} samples found. Using {1:d}'.format(len(self.files), len(self.files)))

        self.__n_samples = len(self.files)

        class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        ObjectLabel.classes = class_names

    def generate(self):
        return self.__generate(self.files)

    def generate_valid(self):
        if not self.test_files:
            print("No valid fraction for test files specified")
        return self.__generate(self.test_files)

    def __generate(self, files):
        current_batch = []
        files_it = iter(files)
        while True:
            try:
                name, label_path, img_path = next(files_it)
                label_file = label_path + name + '.xml'
                current_batch.append(self.__parse_file(label_file, img_path))
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            except StopIteration:
                if self.shuffle: random.shuffle(files)
                files_it = iter(files)

    def __parse_file(self, path, img_directory) -> (Image, ImgLabel):
        with open(path, 'rb') as f:
            tree = ET.parse(f)
            objects = []
            img_file = img_directory + tree.find('filename').text
            img = imread(img_file, color_format=self.color_format)
            for element in tree.findall('object'):
                name = element.find('name').text
                box = element.find('bndbox')
                xmin = int(np.round(float(box.find('xmin').text)))
                ymin = int(np.round(float(box.find('ymin').text)))
                xmax = int(np.round(float(box.find('xmax').text)))
                ymax = int(np.round(float(box.find('ymax').text)))
                objects.append(ObjectLabel(name, [(xmin, img.shape[0] - ymax), (xmax, img.shape[0] - ymin)]))
            return img, ImgLabel(objects), img_file
