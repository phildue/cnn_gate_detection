import numpy as np

from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.refnet import RefNetEncoder
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RefNetPreprocessor(Preprocessor):
    def __init__(self, augmenter: ImgTransform, encoder: RefNetEncoder, n_classes, img_shape, color_format):
        super().__init__(augmenter, encoder, n_classes, img_shape, color_format)

    def preprocess_test(self, dataset: [(Image, ImgLabel)]) -> (np.array, np.array):
        y_batch = []
        x_batch = []
        roi_batch = []
        for img, label, _ in dataset:

            if self.color_format is 'yuv':
                img = img.yuv
            else:
                img = img.bgr

            img, label = resize(img, (self.img_height, self.img_width), label=label)
            #
            # show(img.bgr, t=1)
            img_enc, roi_enc = self.encoder.encode_img(img, label)
            label_enc = self.encoder.encode_label(label, img)
            label_enc = np.expand_dims(label_enc, 0)
            roi_enc = np.expand_dims(roi_enc, 0)
            x_batch.append(img_enc)
            roi_batch.append(roi_enc)
            y_batch.append(label_enc)

        y_batch = np.concatenate(y_batch, 0)
        x_batch = np.concatenate(x_batch, 0)
        roi_batch = np.concatenate(roi_batch, 0)
        return [x_batch, roi_batch], y_batch

    def preprocess(self, img: Image, label: ImgLabel = None):
        if self.color_format is 'yuv':
            img = img.yuv
        else:
            img = img.bgr
        img = resize(img, (self.img_height, self.img_width))
        return self.encoder.encode_img(img, label)

    def preprocess_batch(self, batch: [Image], label: [ImgLabel] = None):
        x_batch = np.zeros((len(batch), self.img_height, self.img_width, 3))
        for i, img in enumerate(batch):
            x_batch[i] = self.preprocess(img, label[i])
        return x_batch
