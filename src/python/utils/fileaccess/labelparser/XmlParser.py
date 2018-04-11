import glob
import xml.etree.ElementTree as ET

import numpy as np

from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.imageprocessing.Backend import imwrite, imread
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class XmlParser(AbstractDatasetParser):

    def write_label(self, label: ImgLabel, path: str):
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = path.replace('xml', self.image_format)
        # TODO extend this with color format and other informations about the dataset
        for obj in label.objects:
            obj_root = ET.SubElement(root, 'object')
            ET.SubElement(obj_root, 'name').text = '{0:s}'.format(obj.class_name)
            bnd_box = ET.SubElement(obj_root, 'bndbox')
            x1 = obj.x_min
            x2 = obj.x_max
            y1 = obj.y_min
            y2 = obj.y_max
            xmin = min((x1, x2))
            xmax = max((x1, x2))
            ymin = min((y1, y2))
            ymax = max((y1, y2))
            ET.SubElement(bnd_box, 'xmin').text = '{0:d}'.format(int(xmin))
            ET.SubElement(bnd_box, 'xmax').text = '{0:d}'.format(int(xmax))
            ET.SubElement(bnd_box, 'ymin').text = '{0:d}'.format(int(ymin))
            ET.SubElement(bnd_box, 'ymax').text = '{0:d}'.format(int(ymax))
            if isinstance(obj, GateLabel):
                pose_root = ET.SubElement(obj_root, 'pose')
                ET.SubElement(pose_root, 'dist_forward').text = '{0:03f}'.format(obj.position.dist_forward)
                ET.SubElement(pose_root, 'dist_side').text = '{0:03f}'.format(obj.position.dist_side)
                ET.SubElement(pose_root, 'lift').text = '{0:03f}'.format(obj.position.lift)
                ET.SubElement(pose_root, 'yaw').text = '{0:03f}'.format(obj.position.yaw)
                ET.SubElement(pose_root, 'pitch').text = '{0:03f}'.format(obj.position.pitch)
                ET.SubElement(pose_root, 'roll').text = '{0:03f}'.format(obj.position.roll)
                corner_root = ET.SubElement(obj_root, 'gate_corners')
                ET.SubElement(corner_root, 'top_left').text = '{0:d},{1:d}'.format(
                    int(obj.gate_corners.top_left[0]),
                    int(obj.gate_corners.top_left[1]))
                ET.SubElement(corner_root, 'top_right').text = '{0:d},{1:d}'.format(
                    int(obj.gate_corners.top_right[0]),
                    int(obj.gate_corners.top_right[1]))
                ET.SubElement(corner_root, 'bottom_left').text = '{0:d},{1:d}'.format(
                    int(obj.gate_corners.bottom_left[0]), int(obj.gate_corners.bottom_left[1]))
                ET.SubElement(corner_root, 'bottom_right').text = '{0:d},{1:d}'.format(
                    int(obj.gate_corners.bottom_right[0]), int(obj.gate_corners.bottom_right[1]))
                ET.SubElement(corner_root, 'center').text = '{0:d},{1:d}'.format(int(obj.gate_corners.center[1]),
                                                                                 int(obj.gate_corners.center[1]))
        tree = ET.ElementTree(root)
        tree.write(path + '.xml')

    def read(self, n=0) -> [Image, ImgLabel]:
        files = sorted(glob.glob(self.directory + '/*.xml'))
        samples = []
        for i, file in enumerate(files):
            if 0 < n < i: break
            label = XmlParser.read_label(file)
            image = imread(file.replace('xml', self.image_format), self.color_format)
            samples.append((image, label))
        return samples

    @staticmethod
    def read_label(path: str) -> [ImgLabel]:
        with open(path, 'rb') as f:
            tree = ET.parse(f)
            objects = []
            for element in tree.findall('object'):
                # TODO extend this to parse gate corners/pose
                name = element.find('name').text
                box = element.find('bndbox')
                x1 = int(np.round(float(box.find('xmin').text)))
                y1 = int(np.round(float(box.find('ymin').text)))
                x2 = int(np.round(float(box.find('xmax').text)))
                y2 = int(np.round(float(box.find('ymax').text)))
                xmin = min((x1, x2))
                xmax = max((x1, x2))
                ymin = min((y1, y2))
                ymax = max((y1, y2))
                label = ObjectLabel(name, np.array([[xmin, ymin], [xmax, ymax]]))
                label.y_min = ymin
                label.y_max = ymax
                label.x_min = xmin
                label.x_max = xmax
                objects.append(label)
            return ImgLabel(objects)
