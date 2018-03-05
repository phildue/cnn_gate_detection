import glob
import pickle

from utils.imageprocessing.Imageprocessing import get_bounding_box
from utils.labels.ImgLabel import ImgLabel


def write_label(path, labels: [ImgLabel], file_format='pkl'):
    LabelFileParser(path, file_format).write(labels)


def read_label(filepath, file_format='pkl'):
    return LabelFileParser(filepath, file_format).read()


class LabelFileParser:
    def __init__(self, directory: str, file_format, start_idx=0):
        self.file_format = file_format
        self.directory = directory
        self.idx = start_idx

    def write(self, labels: [ImgLabel]):
        writers = {'pkl': self.write_label_pkl,
                   'csv': LabelFileParser.write_label_csv,
                   "detection": LabelFileParser.write_label_yolo}
        return writers[self.file_format](self.directory, labels)

    def read(self, n=0) -> [ImgLabel]:
        readers = {'pkl': LabelFileParser.read_label_pkl}

        return readers[self.file_format](self.directory, n)

    @staticmethod
    def write_label_csv(path: str, labels: [ImgLabel]):
        for i, l in enumerate(labels):
            with open(path + '{0:05d}.csv'.format(i), 'w+') as f:
                for l in labels: f.write(l.csv + "\r\n")

    def write_label_pkl(self, path: str, labels: [ImgLabel]):
        for l in labels:
            with open(path + '{0:05d}.pkl'.format(self.idx), 'wb') as f:
                pickle.dump(l, f, pickle.HIGHEST_PROTOCOL)
                self.idx += 1

    @staticmethod
    def write_label_yolo(path: str, labels: [ImgLabel]):
        # <object -class> < x > < y > < width > < height >
        for i, l in enumerate(labels):
            with open(path + '{0:05d}.txt'.format(i), 'w+') as f:
                p1, p2 = get_bounding_box(l.gate_corners)
                f.write("0 " + str(p1[0]) + " " + str(p1[1]) +
                        " " + str(p2[0] - p1[0]) +
                        " " + str(p2[1] - p1[1]))

    @staticmethod
    def read_label_pkl(filepath: str, n=0) -> [ImgLabel]:
        files = sorted(glob.glob(filepath + '/*.pkl'))
        labels = []
        for i, file in enumerate(files):
            if n > 0 and i > n: break
            with open(file, 'rb') as f:
                labels.append(pickle.load(f))
        return labels

    @staticmethod
    def read_label_file_pkl(filepath: str) -> [ImgLabel]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
