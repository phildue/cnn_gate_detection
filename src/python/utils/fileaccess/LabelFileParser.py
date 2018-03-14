import glob
import pickle

import numpy as np

from utils.imageprocessing.Imageprocessing import get_bounding_box
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
import xml.etree.ElementTree as ET

from utils.labels.ObjectLabel import ObjectLabel


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
                   "detection": LabelFileParser.write_label_yolo,
                   'xml': self.write_label_xml}
        return writers[self.file_format](self.directory, labels)

    def read(self, n=0) -> [ImgLabel]:
        readers = {'pkl': LabelFileParser.read_label_pkl,
                   'xml': LabelFileParser.read_label_xml}

        return readers[self.file_format](self.directory, n)

    @staticmethod
    def read_file(file_format, path) -> [ImgLabel]:
        readers = {'pkl': LabelFileParser.read_label_file_pkl,
                   'xml': LabelFileParser.read_label_file_xml}

        return readers[file_format](path)

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

    def write_label_xml(self, path: str, labels: [ImgLabel]):
        for l in labels:
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = '{0:05d}.jpg'.format(self.idx)
            # TODO extend this with color format and other informations about the dataset
            for obj in l.objects:
                obj_root = ET.SubElement(root, 'object')
                ET.SubElement(obj_root, 'name').text = '{0:s}'.format(obj.class_name)
                bnd_box = ET.SubElement(obj_root, 'bndbox')
                ET.SubElement(bnd_box, 'xmin').text = '{0:d}'.format(int(obj.x_min))
                ET.SubElement(bnd_box, 'xmax').text = '{0:d}'.format(int(obj.x_max))
                ET.SubElement(bnd_box, 'ymin').text = '{0:d}'.format(int(obj.y_min))
                ET.SubElement(bnd_box, 'ymax').text = '{0:d}'.format(int(obj.y_max))
                if isinstance(obj, GateLabel):
                    pose_root = ET.SubElement(obj_root, 'pose')
                    ET.SubElement(pose_root, 'dist_forward').text = '{0:03f}'.format(obj.position.dist_forward)
                    ET.SubElement(pose_root, 'dist_side').text = '{0:03f}'.format(obj.position.dist_side)
                    ET.SubElement(pose_root, 'lift').text = '{0:03f}'.format(obj.position.lift)
                    ET.SubElement(pose_root, 'yaw').text = '{0:03f}'.format(obj.position.yaw)
                    ET.SubElement(pose_root, 'pitch').text = '{0:03f}'.format(obj.position.pitch)
                    ET.SubElement(pose_root, 'roll').text = '{0:03f}'.format(obj.position.roll)
                    corner_root = ET.SubElement(obj_root, 'gate_corners')
                    ET.SubElement(corner_root, 'top_left').text = '{0:d},{1:d}'.format(obj.gate_corners.top_left[0],
                                                                                       obj.gate_corners.top_left[1])
                    ET.SubElement(corner_root, 'top_right').text = '{0:d},{1:d}'.format(obj.gate_corners.top_right[0],
                                                                                        obj.gate_corners.top_right[1])
                    ET.SubElement(corner_root, 'bottom_left').text = '{0:d},{1:d}'.format(
                        obj.gate_corners.bottom_left[0], obj.gate_corners.bottom_left[1])
                    ET.SubElement(corner_root, 'bottom_right').text = '{0:d},{1:d}'.format(
                        obj.gate_corners.bottom_right[0], obj.gate_corners.bottom_right[1])
                    ET.SubElement(corner_root, 'center').text = '{0:d},{1:d}'.format(obj.gate_corners.center[1],
                                                                                     obj.gate_corners.center[1])
            tree = ET.ElementTree(root)
            tree.write(path + '/{0:05d}.xml'.format(self.idx))
            self.idx += 1

    @staticmethod
    def read_label_xml(filepath: str, n=0) -> [ImgLabel]:
        files = sorted(glob.glob(filepath + '/*.xml'))
        labels = []
        for i, file in enumerate(files):
            if 0 < n < i: break
            labels.append(LabelFileParser.read_label_file_xml(file))
        return labels

    @staticmethod
    def read_label_file_xml(path: str) -> [ImgLabel]:
        with open(path, 'rb') as f:
            tree = ET.parse(f)
            objects = []
            for element in tree.findall('object'):
                # TODO extend this to parse gate corners/pose
                name = element.find('name').text
                box = element.find('bndbox')
                xmin = int(np.round(float(box.find('xmin').text)))
                ymin = int(np.round(float(box.find('ymin').text)))
                xmax = int(np.round(float(box.find('xmax').text)))
                ymax = int(np.round(float(box.find('ymax').text)))
                label = ObjectLabel(name, [(xmin, ymin), (xmax, ymax)])
                label.y_min = ymin
                label.y_max = ymax
                label.x_min = xmin
                label.x_max = xmax
                objects.append(label)
            return ImgLabel(objects)

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
            if 0 < n < i: break
            with open(file, 'rb') as f:
                labels.append(pickle.load(f))
        return labels

    @staticmethod
    def read_label_file_pkl(filepath: str) -> [ImgLabel]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)