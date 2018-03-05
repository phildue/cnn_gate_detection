from abc import ABC, abstractmethod

import numpy as np

from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class Encoder(ABC):
    @abstractmethod
    def encode_label(self, label: ImgLabel) -> np.array:
        pass

    def encode_label_batch(self, labels: [ImgLabel]) -> np.array:
        labels_enc = []
        for label in labels:
            label_t = self.encode_label(label)
            label_t = np.expand_dims(label_t, 0)
            labels_enc.append(label_t)
        label_t = np.concatenate(labels_enc, 0)
        return label_t

    @abstractmethod
    def encode_img(self, image: Image) -> np.array:
        pass

    def encode_img_batch(self, images: [Image]) -> np.array:
        imgs_enc = []
        for img in images:
            img_t = self.encode_img(img)
            imgs_enc.append(img_t)
        img_t = np.concatenate(imgs_enc, 0)
        return img_t
