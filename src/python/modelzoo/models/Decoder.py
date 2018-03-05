from abc import ABC, abstractmethod

import numpy as np

from src.python.utils import BoundingBox
from src.python.utils.labels.ImgLabel import ImgLabel


class Decoder(ABC):
    @abstractmethod
    def decode_netout_to_label(self, netout) -> ImgLabel:
        pass

    @abstractmethod
    def decode_netout_to_boxes(self, netout) -> [BoundingBox]:
        pass

    @abstractmethod
    def decode_coord(self, coord_t) -> np.array:
        pass

    def decode_netout_to_boxes_batch(self, netout) -> [BoundingBox]:
        boxes = []
        for i in range(netout.shape[0]):
            boxes_i = self.decode_netout_to_boxes(netout[i])
            boxes.append(boxes_i)

        return boxes

    def decode_netout_to_label_batch(self, netout) -> [BoundingBox]:
        labels = []
        for i in range(netout.shape[0]):
            boxes = self.decode_netout_to_boxes(netout[i])
            label = BoundingBox.to_label(boxes)
            labels.append(label)

        return labels
