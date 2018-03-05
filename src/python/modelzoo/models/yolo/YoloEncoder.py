import numpy as np
from frontend.models.Encoder import Encoder
from imageprocessing.Backend import normalize
from labels.ImgLabel import ImgLabel

from src.python.utils.imageprocessing.Image import Image


class YoloEncoder(Encoder):
    def __init__(self, img_norm=(416, 416), grid=(13, 13), n_boxes=5, n_classes=20,
                 color_format='yuv'):
        self.color_format = color_format
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.grid = grid
        self.norm = img_norm

    def encode_img(self, image: Image):
        img = normalize(image)
        return np.expand_dims(img.array, axis=0)

    def encode_label(self, label: ImgLabel):
        """
        Converts label to y vector how it is used for yolo learning.
        Each object in the image gets assigned to the anchor boxes.

        :param label: image label containing objects, their bounding boxes and names
        :return: y-tensor usually with shape (13,13,5,25)
         where 13x13 is the grid, each grid cell contains 5 anchor boxes
         each anchor box contains:
            (x,y) as center of the box (offset from grid cell, normalized with respect to image size)
            (w,h) as width and height (normalized with respect to image size)
            (c) confidence that there is an object
            (cl)*20 class likelihoods

        """
        y = np.zeros((self.grid[0], self.grid[1], self.n_boxes, 5 + self.n_classes))
        for obj in label.objects:
            w = obj.x_max - obj.x_min
            h = obj.y_max - obj.y_min

            center_x = obj.x_min + w / 2
            center_x = center_x / (float(self.norm[1]) / self.grid[1])
            center_y = obj.y_min + h / 2
            center_y = self.norm[1] - center_y  # flip because we use different coordinate system
            center_y = center_y / (float(self.norm[0]) / self.grid[0])

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            grid_x = np.maximum(np.minimum(grid_x, self.grid[0] - 1), 0)
            grid_y = np.maximum(np.minimum(grid_y, self.grid[1] - 1), 0)

            obj_idx = obj.class_id
            w /= (float(self.norm[1]) / self.grid[1])
            h /= (float(self.norm[0]) / self.grid[0])
            box = [center_x - grid_x, center_y - grid_y, w, h]

            y[grid_y, grid_x, :, 0:4] = self.n_boxes * [box]
            y[grid_y, grid_x, :, 4] = self.n_boxes * [1.]
            y[grid_y, grid_x, :, 5:] = self.n_boxes * [[0.] * self.n_classes]
            y[grid_y, grid_x, :, 5 + obj_idx] = 1.0

        return y.reshape([self.grid[1], self.grid[0], self.n_boxes, (self.n_classes + 5)])
