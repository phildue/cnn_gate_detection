from random import randint

import numpy as np

from modelzoo.models.Encoder import Encoder
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Backend import normalize, crop
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.labels.ImgLabel import ImgLabel


class RefNetEncoder(Encoder):
    def __init__(self, anchor_dims=None, img_norm=(416, 416), grids=None, n_boxes=5, n_polygon=4,
                 color_format='yuv', n_regions=5):
        self.n_regions = n_regions
        if anchor_dims is None:
            anchor_dims = [np.array([[1.08, 1.19],
                                     [3.42, 4.41],
                                     [6.63, 11.38],
                                     [9.42, 5.11],
                                     [16.62, 10.52]])]
        if grids is None:
            grids = [(13, 13)]

        self.anchor_dims = anchor_dims
        self.n_polygon = n_polygon
        self.color_format = color_format
        self.n_boxes = n_boxes
        self.grids = grids
        self.norm = img_norm
        self.encoder = GateNetEncoder(anchor_dims, (52, 52), grids, n_boxes, n_polygon, color_format)

    def _extract_roi(self, img: Image, label: ImgLabel):
        rois = np.zeros((self.n_regions, 4))
        label_rois = []
        areas = np.array([obj.area for obj in label.objects])
        order = list(reversed(np.argsort(areas)))
        filtered_objs = []
        if len(areas) > 0:
            for i in range(np.min((self.n_regions, len(areas)))):
                filtered_objs.append(label.objects[order[i]])

        for i, obj in enumerate(filtered_objs):

            w = 1.25 * obj.width
            h = 1.25 * obj.height
            if w > h:
                h += w - h
            elif h > w:
                w += h - w

            crop_min = (max(0, obj.cx - w / 2)), max(obj.cy - h / 2, 0),
            crop_max = min(obj.cx + w / 2, self.norm[1]), min(obj.cy + h / 2, self.norm[0])
            print(obj)
            img_crop, label_crop = crop(img, crop_min, crop_max, label)
            print("Crop Min", crop_min)
            print("Crop Max", crop_max)
            print(label_crop)
            show(img_crop, labels=label_crop, t=1, name='crop')
            if img_crop.array.size > 0:
                rois[i] = crop_min[0], crop_min[1], img_crop.shape[1], img_crop.shape[0]
                label_rois.append(label_crop)

        return rois, label_rois

    def encode_img(self, image: Image, label: ImgLabel = None):
        rois, label_rois = self._extract_roi(image, label)
        return np.expand_dims(image.array, axis=0), rois

    def encode_label(self, label: ImgLabel, img=None):
        """
        Encodes bounding box in ground truth tensor.

        :param label: image label containing objects, their bounding boxes and names
        :return: label-tensor

        """
        rois, label_rois = self._extract_roi(img, label)
        label_t = []
        for i in range(self.n_regions):
            l = label_rois[i] if i < len(label_rois) else ImgLabel([])
            label_enc = self.encoder.encode_label(l)
            roi_enc = np.tile(rois[i], (label_enc.shape[0], 1))
            label_enc = np.hstack((label_enc, roi_enc))
            label_enc = np.expand_dims(label_enc, 0)
            label_t.append(label_enc)
        label_t = np.concatenate(label_t, 0)
        label_t = np.reshape(label_t, (-1, self.n_polygon + 1 + 4 + 4))
        return label_t

    def encode_label_batch(self, labels: [ImgLabel], imgs: [Image] = None) -> np.array:
        labels_enc = []
        for i, label in enumerate(labels):
            label_t = self.encode_label(label, imgs[i])
            label_t = np.expand_dims(label_t, 0)
            labels_enc.append(label_t)
        label_t = np.concatenate(labels_enc, 0)
        return label_t

    def encode_img_batch(self, images: [Image], labels: [ImgLabel] = None) -> np.array:
        imgs_enc = []
        for i, img in enumerate(images):
            img_t = self.encode_img(img, labels[i])
            imgs_enc.append(img_t)
        img_t = np.concatenate(imgs_enc, 0)
        return img_t
