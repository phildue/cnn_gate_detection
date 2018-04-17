import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class GateNetEncoder(Encoder):
    def __init__(self, img_norm=(416, 416), grid=(13, 13), n_boxes=5, n_polygon=4,
                 color_format='yuv'):
        self.n_polygon = n_polygon
        self.color_format = color_format
        self.n_boxes = n_boxes
        self.grid = grid
        self.norm = img_norm

    def encode_img(self, image: Image):
        # TODO do we need this normalization?
        img = normalize(image)
        return np.expand_dims(img.array, axis=0)

    def encode_label(self, label: ImgLabel):
        """
        Encodes bounding box in ground truth tensor.

        :param label: image label containing objects, their bounding boxes and names
        :return: label-tensor

        """
        label_t = np.zeros((self.grid[0], self.grid[1], self.n_boxes, self.n_polygon + 1))
        for obj in label.objects:
            w = obj.x_max - obj.x_min
            h = obj.y_max - obj.y_min

            center_x = obj.x_min + w / 2
            center_x = center_x / (float(self.norm[1]) / self.grid[1])
            center_y = obj.y_min + h / 2
            center_y = self.norm[0] - center_y  # flip because we use different coordinate system
            center_y = center_y / (float(self.norm[0]) / self.grid[0])

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            grid_x = np.maximum(np.minimum(grid_x, self.grid[1] - 1), 0)
            grid_y = np.maximum(np.minimum(grid_y, self.grid[0] - 1), 0)

            w /= (float(self.norm[1]) / self.grid[1])
            h /= (float(self.norm[0]) / self.grid[0])
            cx = center_x - grid_x
            cy = center_y - grid_y
            box = [cx, cy, w, h]

            # TODO previous objects are overwritten instead we should assign the anchor that has the highes iou
            # and thus allow multiple true objects per grid
            label_t[grid_y, grid_x, :, 0:4] = self.n_boxes * [box]
            label_t[grid_y, grid_x, :, 4] = self.n_boxes * [1.]
            label_t[grid_y, grid_x, :, 5:] = self.n_boxes * [[0.] * 1]
            label_t[grid_y, grid_x, :, 5 + obj.class_id] = 1.0

        label_t = np.reshape(label_t, [self.grid[0] * self.grid[1] * self.n_boxes, 1 + self.n_polygon])

        return label_t
