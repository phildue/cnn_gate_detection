import random

import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel


class RandomRotate(ImgTransform):

    def __init__(self, ang_min=-30.0, ang_max=30.0):
        self.ang_min = ang_min
        self.ang_max = ang_max

    def transform(self, img: Image, label: ImgLabel):
        rows, cols, _ = img.shape
        img_aug = img.copy()
        label_aug = label.copy()
        angle = random.uniform(self.ang_min, self.ang_max)

        rotmat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # FIXME this is not working
        for obj in label_aug.objects:
            box = np.array([[obj.x_min, obj.x_max],
                            [obj.y_min, obj.y_max],
                            [1, 1]])

            box_rot = rotmat.dot(box)
            obj.x_min = np.min(box_rot[0, :])
            obj.x_max = np.max(box_rot[0, :])
            obj.y_min = np.min(box_rot[1, :])
            obj.y_max = np.max(box_rot[1, :])
            # if isinstance(obj, GateLabel):
            #     gate_corners = obj.gate_corners
            #     gate_corners -= center
            #     gate_corners = rotmat.dot(obj.gate_corners.as_mat)
            #     gate_corners += center

        img_aug.array = cv2.warpAffine(img_aug.array, rotmat, (cols, rows))

        return img_aug, label_aug
