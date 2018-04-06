import sys

import cv2
import numpy as np
from math import isnan

from utils.imageprocessing.Backend import convert_color, COLOR_RGBA2BGR
from utils.imageprocessing.Image import Image
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.Pose import Pose

sys.path.extend(["c:/Users/mail-/Documents/code/dronerace2018/target/simulator/AirSim/PythonClient"])
from AirSimClient import *


class AirSimClient:

    @staticmethod
    def _response2mat(response):
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(response.height, response.width, 4)
        return img_rgba

    @staticmethod
    def _segment2corner(segment_labels: Image, label_color):
        coordinates = np.where(np.all(segment_labels == label_color, -1))
        corner = np.mean(coordinates, -1)
        return corner

    def retrieve_samples(self) -> (Image, ImgLabel):
        responses = self.client.simGetImages([
            ImageRequest(0, AirSimImageType.Segmentation, False, False),
            ImageRequest(0, AirSimImageType.Scene, False, False)])

        seg = self._response2mat(responses[0])
        scene = self._response2mat(responses[1])
        scene = convert_color(Image(scene, 'bgr'), COLOR_RGBA2BGR)
        hcam, wcam = scene.shape[:2]
        hseg, wseg = seg.shape[:2]
        hscale = hcam / hseg
        wscale = wcam / wseg
        scale = np.array([hscale, wscale])
        gate_labels = []
        for i in range(1, self.n_gates + 1, 4):
            bottom_left = self._segment2corner(seg, np.array(self._color_lookup[i])) * scale
            top_left = self._segment2corner(seg, np.array(self._color_lookup[i + 1])) * scale
            top_right = self._segment2corner(seg, np.array(self._color_lookup[i + 2])) * scale
            bottom_right = self._segment2corner(seg, np.array(self._color_lookup[i + 3])) * scale
            center = np.array([(np.abs(top_left[0] + bottom_left[0])) / 2, np.abs(top_left[1] + top_right[1]) / 2])
            print(bottom_left)
            # TODO what if one corner is missing
            if np.isnan(bottom_left).any() or \
                    np.isnan(top_left).any() or \
                    np.isnan(top_right).any() or \
                    np.isnan(bottom_right).any():
                continue
            gate_label = GateLabel(position=Pose(), gate_corners=GateCorners((center[1], hcam - center[0]),
                                                                             (top_left[1], hcam - top_left[0]),
                                                                             (top_right[1], hcam - top_right[0]),
                                                                             (bottom_left[1], hcam - bottom_left[0]),
                                                                             (bottom_right[1], hcam - bottom_right[0])))
            gate_labels.append(gate_label)

        return scene, ImgLabel(gate_labels)

    def __init__(self, n_gates, address=None):
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.simSetSegmentationObjectID("[\w]*", 0, True)

        for i in range(1, n_gates + 1, 4):
            bl = self.client.simSetSegmentationObjectID("bl" + str(i), i)
            tl = self.client.simSetSegmentationObjectID("tl" + str(i), i + 1)
            tr = self.client.simSetSegmentationObjectID("tr" + str(i), i + 2)
            br = self.client.simSetSegmentationObjectID("br" + str(i), i + 3)
            if not (br and tl and tr and bl):
                print("AirSimClient::Warning! at least one corner was not found")

        self.n_gates = n_gates

        self._color_lookup = np.array([[55, 181, 57],
                                       [153, 108, 6],
                                       [112, 105, 191],
                                       [89, 121, 72],
                                       [190, 225, 64],
                                       [206, 190, 59],
                                       [81, 13, 36],
                                       [115, 176, 195],
                                       [161, 171, 27],
                                       [135, 169, 180],
                                       [29, 26, 199],
                                       [102, 16, 239],
                                       [242, 107, 146],
                                       [156, 198, 23],
                                       [49, 89, 160],
                                       [68, 218, 116],
                                       [11, 236, 9],
                                       [196, 30, 8],
                                       [121, 67, 28],
                                       [0, 53, 65],
                                       [146, 52, 70],
                                       [226, 149, 143],
                                       [151, 126, 171],
                                       [194, 39, 7],
                                       [205, 120, 161],
                                       [212, 51, 60],
                                       [211, 80, 208],
                                       [189, 135, 188],
                                       [54, 72, 205],
                                       [103, 252, 157],
                                       [124, 21, 123],
                                       [19, 132, 69],
                                       [195, 237, 132],
                                       [94, 253, 175],
                                       [182, 251, 87],
                                       [90, 162, 242],
                                       [199, 29, 1],
                                       [254, 12, 229],
                                       [35, 196, 244],
                                       [220, 163, 49],
                                       [86, 254, 214],
                                       [152, 3, 129],
                                       [92, 31, 106],
                                       [207, 229, 90],
                                       [125, 75, 48],
                                       [98, 55, 74],
                                       [126, 129, 238],
                                       [222, 153, 109],
                                       [85, 152, 34],
                                       [173, 69, 31],
                                       [37, 128, 125],
                                       [58, 19, 33],
                                       [134, 57, 119],
                                       [218, 124, 115],
                                       [120, 0, 200],
                                       [225, 131, 92],
                                       [246, 90, 16],
                                       [51, 155, 241],
                                       [202, 97, 155],
                                       [184, 145, 182],
                                       [96, 232, 44],
                                       [133, 244, 133],
                                       [180, 191, 29],
                                       [1, 222, 192],
                                       [99, 242, 104],
                                       [91, 168, 219],
                                       [65, 54, 217],
                                       [148, 66, 130],
                                       [203, 102, 204],
                                       [216, 78, 75],
                                       [234, 20, 250],
                                       [109, 206, 24],
                                       [164, 194, 17],
                                       [157, 23, 236],
                                       [158, 114, 88],
                                       [245, 22, 110],
                                       [67, 17, 35],
                                       [181, 213, 93],
                                       [170, 179, 42],
                                       [52, 187, 148],
                                       [247, 200, 111],
                                       [25, 62, 174],
                                       [100, 25, 240],
                                       [191, 195, 144],
                                       [252, 36, 67],
                                       [241, 77, 149],
                                       [237, 33, 141],
                                       [119, 230, 85],
                                       [28, 34, 108],
                                       [78, 98, 254],
                                       [114, 161, 30],
                                       [75, 50, 243],
                                       [66, 226, 253],
                                       [46, 104, 76],
                                       [8, 234, 216],
                                       [15, 241, 102],
                                       [93, 14, 71],
                                       [192, 255, 193],
                                       [253, 41, 164],
                                       [24, 175, 120],
                                       [185, 243, 231],
                                       [169, 233, 97],
                                       [243, 215, 145],
                                       [72, 137, 21],
                                       [160, 113, 101],
                                       [214, 92, 13],
                                       [167, 140, 147],
                                       [101, 109, 181],
                                       [53, 118, 126],
                                       [3, 177, 32],
                                       [40, 63, 99],
                                       [186, 139, 153],
                                       [88, 207, 100],
                                       [71, 146, 227],
                                       [236, 38, 187],
                                       [215, 4, 215],
                                       [18, 211, 66],
                                       [113, 49, 134],
                                       [47, 42, 63],
                                       [219, 103, 127],
                                       [57, 240, 137],
                                       [227, 133, 211],
                                       [145, 71, 201],
                                       [217, 173, 183],
                                       [250, 40, 113],
                                       [208, 125, 68],
                                       [224, 186, 249],
                                       [69, 148, 46],
                                       [239, 85, 20],
                                       [108, 116, 224],
                                       [56, 214, 26],
                                       [179, 147, 43],
                                       [48, 188, 172],
                                       [221, 83, 47],
                                       [155, 166, 218],
                                       [62, 217, 189],
                                       [198, 180, 122],
                                       [201, 144, 169],
                                       [132, 2, 14],
                                       [128, 189, 114],
                                       [163, 227, 112],
                                       [45, 157, 177],
                                       [64, 86, 142],
                                       [118, 193, 163],
                                       [14, 32, 79],
                                       [200, 45, 170],
                                       [74, 81, 2],
                                       [59, 37, 212],
                                       [73, 35, 225],
                                       [95, 224, 39],
                                       [84, 170, 220],
                                       [159, 58, 173],
                                       [17, 91, 237],
                                       [31, 95, 84],
                                       [34, 201, 248],
                                       [63, 73, 209],
                                       [129, 235, 107],
                                       [231, 115, 40],
                                       [36, 74, 95],
                                       [238, 228, 154],
                                       [61, 212, 54],
                                       [13, 94, 165],
                                       [141, 174, 0],
                                       [140, 167, 255],
                                       [117, 93, 91],
                                       [183, 10, 186],
                                       [165, 28, 61],
                                       [144, 238, 194],
                                       [12, 158, 41],
                                       [76, 110, 234],
                                       [150, 9, 121],
                                       [142, 1, 246],
                                       [230, 136, 198],
                                       [5, 60, 233],
                                       [232, 250, 80],
                                       [143, 112, 56],
                                       [187, 70, 156],
                                       [2, 185, 62],
                                       [138, 223, 226],
                                       [122, 183, 222],
                                       [166, 245, 3],
                                       [175, 6, 140],
                                       [240, 59, 210],
                                       [248, 44, 10],
                                       [83, 82, 52],
                                       [223, 248, 167],
                                       [87, 15, 150],
                                       [111, 178, 117],
                                       [197, 84, 22],
                                       [235, 208, 124],
                                       [9, 76, 45],
                                       [176, 24, 50],
                                       [154, 159, 251],
                                       [149, 111, 207],
                                       [168, 231, 15],
                                       [209, 247, 202],
                                       [80, 205, 152],
                                       [178, 221, 213],
                                       [27, 8, 38],
                                       [244, 117, 51],
                                       [107, 68, 190],
                                       [23, 199, 139],
                                       [171, 88, 168],
                                       [136, 202, 58],
                                       [6, 46, 86],
                                       [105, 127, 176],
                                       [174, 249, 197],
                                       [172, 172, 138],
                                       [228, 142, 81],
                                       [7, 204, 185],
                                       [22, 61, 247],
                                       [233, 100, 78],
                                       [127, 65, 105],
                                       [33, 87, 158],
                                       [139, 156, 252],
                                       [42, 7, 136],
                                       [20, 99, 179],
                                       [79, 150, 223],
                                       [131, 182, 184],
                                       [110, 123, 37],
                                       [60, 138, 96],
                                       [210, 96, 94],
                                       [123, 48, 18],
                                       [137, 197, 162],
                                       [188, 18, 5],
                                       [39, 219, 151],
                                       [204, 143, 135],
                                       [249, 79, 73],
                                       [77, 64, 178],
                                       [41, 246, 77],
                                       [16, 154, 4],
                                       [116, 134, 19],
                                       [4, 122, 235],
                                       [177, 106, 230],
                                       [21, 119, 12],
                                       [104, 5, 98],
                                       [50, 130, 53],
                                       [30, 192, 25],
                                       [26, 165, 166],
                                       [10, 160, 82],
                                       [106, 43, 131],
                                       [44, 216, 103],
                                       [255, 101, 221],
                                       [32, 151, 196],
                                       [213, 220, 89],
                                       [70, 209, 228],
                                       [97, 184, 83],
                                       [82, 239, 232],
                                       [251, 164, 128],
                                       [193, 11, 245],
                                       [38, 27, 159],
                                       [229, 141, 203],
                                       [130, 56, 55],
                                       [147, 210, 11],
                                       [162, 203, 118],
                                       [3, 47, 206],
                                       ])
        self._color_lookup = np.hstack([self._color_lookup, np.ones((self._color_lookup.shape[0], 1)) * 255.0])
