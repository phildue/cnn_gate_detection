import glob
import xml.etree.ElementTree as ET

import numpy as np

from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.imageprocessing.Backend import imread
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Pose import Pose


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
                ET.SubElement(pose_root, 'north').text = '{0:03f}'.format(obj.pose.north)
                ET.SubElement(pose_root, 'east').text = '{0:03f}'.format(obj.pose.east)
                ET.SubElement(pose_root, 'up').text = '{0:03f}'.format(obj.pose.up)
                ET.SubElement(pose_root, 'yaw').text = '{0:03f}'.format(obj.pose.yaw)
                ET.SubElement(pose_root, 'pitch').text = '{0:03f}'.format(obj.pose.pitch)
                ET.SubElement(pose_root, 'roll').text = '{0:03f}'.format(obj.pose.roll)
                corner_root = ET.SubElement(obj_root, 'gate_corners')
                ET.SubElement(corner_root, 'top_left').text = '{},{}'.format(
                    int(obj.gate_corners.top_left[0]),
                    int(obj.gate_corners.top_left[1]))
                ET.SubElement(corner_root, 'top_right').text = '{},{}'.format(
                    int(obj.gate_corners.top_right[0]),
                    int(obj.gate_corners.top_right[1]))
                ET.SubElement(corner_root, 'bottom_left').text = '{},{}'.format(
                    int(obj.gate_corners.bottom_left[0]), int(obj.gate_corners.bottom_left[1]))
                ET.SubElement(corner_root, 'bottom_right').text = '{},{}'.format(
                    int(obj.gate_corners.bottom_right[0]), int(obj.gate_corners.bottom_right[1]))
                ET.SubElement(corner_root, 'center').text = '{},{}'.format(int(obj.gate_corners.center[0]),
                                                                           int(obj.gate_corners.center[1]))
        tree = ET.ElementTree(root)
        tree.write(path + '.xml')

    def read(self, n=0) -> ([Image], [ImgLabel]):
        files = sorted(glob.glob(self.directory + '/*.xml'))
        samples = []
        labels = []
        for i, file in enumerate(files):
            if n > 0 and 0 < n < i: break
            label = XmlParser.read_label(file)
            if label is None: continue
            image = imread(file.replace('xml', self.image_format), self.color_format)
            samples.append(image)
            labels.append(label)
        return samples, labels

    @staticmethod
    def _parse_gate_corners(gate_corners_xml: str) -> GateCorners:
        top_left = tuple([int(e) for e in gate_corners_xml.find('top_left').text.split(',')])
        top_right = tuple([int(e) for e in gate_corners_xml.find('top_right').text.split(',')])
        bottom_left = tuple([int(e) for e in gate_corners_xml.find('bottom_left').text.split(',')])
        bottom_right = tuple([int(e) for e in gate_corners_xml.find('bottom_right').text.split(',')])
        center = (np.array(bottom_left) + np.array(top_right))/2

        gate_corners = GateCorners(center=center, top_left=top_left, top_right=top_right,
                                   bottom_left=bottom_left, bottom_right=bottom_right)
        return gate_corners

    @staticmethod
    def _parse_pose(pose_xml: str) -> Pose:
        north = float(pose_xml.find('north').text)
        east = float(pose_xml.find('east').text)
        up = float(pose_xml.find('down').text)
        yaw = float(pose_xml.find('yaw').text)
        pitch = float(pose_xml.find('pitch').text)
        roll = float(pose_xml.find('roll').text)

        pose = Pose(north, east, up, roll, pitch, yaw)
        return pose

    @staticmethod
    def read_label(path: str) -> [ImgLabel]:
        with open(path, 'rb') as f:
            try:
                tree = ET.parse(f)
                objects = []
                for element in tree.findall('object'):
                    name = element.find('name').text

                    gate_corners_xml = element.find('gate_corners')
                    if gate_corners_xml:

                        gate_corners = XmlParser._parse_gate_corners(gate_corners_xml)
                        try:
                            pose_xml = element.find('pose')
                            pose = XmlParser._parse_pose(pose_xml)
                        except AttributeError:
                            pose = Pose()

                        label = GateLabel(pose, gate_corners)

                    else:
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

                    confidence_field = element.find('confidence')
                    if confidence_field is not None:
                        confidence = float(confidence_field.text)
                    else:
                        confidence = 1.0
                    label.confidence = confidence
                    objects.append(label)
                return ImgLabel(objects)
            except ET.ParseError:
                print('Error parsing: ' + path)
