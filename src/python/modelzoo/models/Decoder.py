from abc import ABC, abstractmethod

import numpy as np

from utils.labels.ImgLabel import ImgLabel


class Decoder(ABC):
    @abstractmethod
    def decode_netout(self, netout) -> ImgLabel:
        pass

    @abstractmethod
    def decode_coord(self, coord_t) -> np.array:
        pass

    def decode_netout_batch(self, netout) -> [ImgLabel]:
        labels = []
        for i in range(netout.shape[0]):
            label = self.decode_netout(netout[i])
            labels.append(label)

        return labels
