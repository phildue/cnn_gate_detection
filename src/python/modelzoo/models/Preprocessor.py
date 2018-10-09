import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.imageprocessing.Backend import resize, crop
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class Preprocessor:
    def __init__(self, augmenter: ImgTransform, encoder: Encoder, n_classes, img_shape, color_format,
                 preprocess_transformer: ImgTransform = None):
        self.preprocess_transformer = preprocess_transformer
        self.color_format = color_format
        self.img_height, self.img_width = img_shape[:2]
        self.n_classes = n_classes
        self.encoder = encoder
        self.augmenter = augmenter

    def preprocess_train_generator(self, batches: [[(Image, ImgLabel)]]):
        for batch in batches:
            yield self.preprocess_train(batch)

    def preproces_test_generator(self, batches: [[(Image, ImgLabel)]]):
        for batch in batches:
            yield self.preprocess_test(batch)

    def preprocess_test(self, dataset: [(Image, ImgLabel)]) -> (np.array, np.array):
        y_batch = []
        x_batch = []
        for img, label, _ in dataset:

            if self.color_format is not img.format:
                raise ValueError("Invalid Color Format")

            if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
                img, label = self.crop_to_input(img, label)
                img, label = resize(img, (self.img_height, self.img_width), label=label)
            #
            # show(img.bgr,labels=label, t=0)

            img_enc = self.encoder.encode_img(img)
            label_enc = self.encoder.encode_label(label)
            label_enc = np.expand_dims(label_enc, 0)
            x_batch.append(img_enc)
            y_batch.append(label_enc)

        y_batch = np.concatenate(y_batch, 0)
        x_batch = np.concatenate(x_batch, 0)
        return x_batch, y_batch

    def preprocess_train(self, dataset: [(Image, ImgLabel)]) -> (np.array, np.array):
        dataset_augmented = []

        if self.augmenter is not None:
            for img, label, path in dataset:
                img, label = self.augmenter.transform(img, label)
                dataset_augmented.append((img, label, path))
        else:
            dataset_augmented = dataset

        return self.preprocess_test(dataset_augmented)

    def preprocess(self, img: Image):

        if self.preprocess_transformer is not None:
            img, _ = self.preprocess_transformer.transform(img, ImgLabel([]))

        if self.color_format is 'yuv':
            img = img.yuv
        else:
            img = img.bgr

        if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
            img = self.crop_to_input(img)
            img = resize(img, (self.img_height, self.img_width))
        return self.encoder.encode_img(img)

    def preprocess_batch(self, batch: [Image]):
        x_batch = np.zeros((len(batch), self.img_height, self.img_width, 3))
        for i, img in enumerate(batch):
            x_batch[i] = self.preprocess(img)
        return x_batch

    def crop_to_input(self, img, label=None):
        """
        Crop the center part
        :param img:
        :param label:
        :return:
        """
        shape = np.array(img.shape[:2])
        short_side = np.argmin(shape)
        diff = np.max(shape) - np.min(shape)
        diff2 = diff / 2

        if short_side == 0:
            return crop(img, (diff2, 0), (shape[1] - diff2, shape[0]), label)

        elif short_side == 1:
            return crop(img, (0, diff2), (shape[1], shape[0] - diff2), label)

        else:
            raise ValueError("Short side doesnt make sence")

    def minmax(self, dataset):
        img_t = np.concatenate([d[0].array for d in dataset], 0)
        d_min = np.min(img_t, -1)
        d_max = np.max(img_t, -1)
        return d_min, d_max
