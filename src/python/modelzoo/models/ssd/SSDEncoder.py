from enum import Enum

import numpy as np


# noinspection PyDefaultArgument
# noinspection PyDefaultArgument
from modelzoo.backend.tensor.iou import iou_np
from modelzoo.models.Encoder import Encoder
from utils.BoundingBox import centroid_to_minmax, BoundingBox
from utils.imageprocessing import Image
from utils.labels.ImgLabel import ImgLabel


class Label(Enum):
    BACKGROUND = -2
    FOREGROUND = -1


class SSDEncoder(Encoder):
    """
    A class to transform ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an ssd model, and to transform predictions of the ssd model back
    to the original format of the input labels.
    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    """

    def __init__(self, img_shape, n_classes, anchors_t, variances=[1.0, 1.0, 1.0, 1.0], iou_thresh=0.5,
                 neg_iou_thresh=0.2):

        self.neg_iou_thresh = neg_iou_thresh
        self.anchors_t = anchors_t

        self.img_height = img_shape[0]
        self.img_width = img_shape[1]
        self.n_classes = n_classes

        self.variances = variances
        self.iou_thresh = iou_thresh

    def encode_img(self, image: Image):
        return np.expand_dims(image.array, axis=0)

    def _match_to_anchors(self, true_boxes_t):
        """
        Determines which anchor boxes are responsible for which GT box
        :param true_boxes_t: tensor(#true boxes, 4) coordinates of true boxes in minmax format as tensor
        :return: list(#boxes) The idx which anchor box is assigned to which true box -1 if responsible for background
        """

        anchors_minmax = centroid_to_minmax(self.anchors_t)

        iou = iou_np(true_boxes_t, anchors_minmax)

        # Assign each anchor the true box its responsible for
        # anchors that don't have sufficient iou get BACKGROUND
        # anchors that are somewhere in between get FOREGROUND
        # matches have the corresponding gt indices
        anchor_to_truth_iou = np.max(iou, 0)
        anchor_to_truth_idx = np.argmax(iou, 0)
        anchor_to_truth_idx[anchor_to_truth_iou < self.neg_iou_thresh] = Label.BACKGROUND.value
        anchor_to_truth_idx[(self.neg_iou_thresh <= anchor_to_truth_iou) & (
            anchor_to_truth_iou < self.iou_thresh)] = Label.FOREGROUND.value

        # Get the best anchor for each truth
        best_anchor_idx = np.zeros((iou.shape[0],), dtype=np.int)
        iou_copy = iou.copy()
        for i in range(iou.shape[0]):
            best_anchor_idx[i] = int(np.argmax(iou_copy[i, :]))
            # We don't want to select the same anchor twice
            iou_copy[:, best_anchor_idx[i]] = 0

        # Overwrite the original responsibility to make sure
        # most true boxes get an anchor (even when 0 < iou < thresh)
        for j in range(best_anchor_idx.shape[0]):
            if anchor_to_truth_iou[best_anchor_idx[j]] > 0:
                anchor_to_truth_idx[best_anchor_idx[j]] = j

        return anchor_to_truth_idx

    def _generate_coord_t(self, matches_idx, true_boxes: [BoundingBox]):
        """
        Generates box coordinate tensor based on true boxes
        :param matches_idx: tensor(#boxes,#true_boxes) Assignment between anchors and true boxes
        :param true_boxes: Bounding boxes that contain box coordinates
        :return: tensor (#boxes,4)
        """

        coord_t = self.anchors_t.copy()
        for i, match_idx in enumerate(matches_idx):
            if match_idx > Label.FOREGROUND.value:
                matched_box = true_boxes[match_idx]
                coord_t[i] = matched_box.coords_centroid

                # normalize to image size
        coord_t[:, 0] /= self.img_width
        coord_t[:, 2] /= self.img_width
        coord_t[:, 1] /= self.img_height
        coord_t[:, 3] /= self.img_height

        # normalize to anchor box
        anchor_wh = self.anchors_t[:, 2:] / np.array([self.img_width, self.img_height])
        anchor_cxy = self.anchors_t[:, :2] / np.array([self.img_width, self.img_height])

        coord_t[:, :2] = coord_t[:, :2] - anchor_cxy
        coord_t[:, 0] = coord_t[:, 0] / (anchor_wh[:, 0] * self.variances[0])
        coord_t[:, 1] = coord_t[:, 1] / (anchor_wh[:, 1] * self.variances[1])
        coord_t[:, 2] = coord_t[:, 2] / anchor_wh[:, 0]
        coord_t[:, 3] = coord_t[:, 3] / anchor_wh[:, 1]

        # width and height are encoded in logarithm
        # although this should be impossible we restrict it to a positive value above 0
        w_enc = np.log(coord_t[:, 2])
        h_enc = np.log(coord_t[:, 3])

        coord_t[:, 2] = w_enc / self.variances[2]
        coord_t[:, 3] = h_enc / self.variances[3]

        return coord_t

    def _generate_coord_t_empty(self):
        """
        Generates box coordinate tensor when there are no objects in the image
        :return: tensor(#boxes,4)
        """
        return np.zeros((self.anchors_t.shape[0], 4))

    def _generate_class_t(self, matches_idx, true_boxes: [BoundingBox]):
        """
        Generates a tensor containing the class labels where idx 0 is the background class
        :param matches_iou: tensor(#boxes,#true_boxes) The iou overlap of the matches between overlap and true boxes
        :param matches_idx: tensor(#boxes,#true_boxes) The idx which anchor box is assigned to which true box
        :param true_boxes: Bounding boxes with GT data
        :return: tensor(#boxes,#classes)
        """
        class_coding = np.eye(self.n_classes)
        classes_t = np.zeros((self.anchors_t.shape[0], self.n_classes))

        for i, match_idx in enumerate(matches_idx):
            if match_idx > Label.FOREGROUND.value:
                matched_box = true_boxes[match_idx]
                classes_t[i] = class_coding[matched_box.prediction + 1]
            elif match_idx == Label.BACKGROUND.value:
                classes_t[i] = class_coding[0]

        return classes_t

    def _generate_class_t_empty(self):
        """
        Generates a tensor containing the class labels when there is no object in the image
        :return: tensor(#boxes,#classes)
        """
        class_coding = np.eye(self.n_classes)
        classes_t = np.zeros((self.anchors_t.shape[0], self.n_classes))
        classes_t += class_coding[0]
        return classes_t

    def encode_label(self, label: ImgLabel):
        """
        Encode an image label to tensor as required for ssd. The empty anchor boxes are generated,
        based on the model architecture. Each anchor box gets assigned a ground truth box; then contains
        the relative GT coordinates and the one hot encoded class label. Anchor boxes that don't have
        sufficient overlap with any GT box are labeled as background that is class label are all zeros a part
        from 0 which is 1
        :param label: label of image
        :return tensor(#boxes,#classes + 4) containing label tensor as required for ssd

        """
        true_boxes = BoundingBox.from_label(label)

        if len(true_boxes) > 0:
            true_boxes_t = BoundingBox.to_tensor_minmax(true_boxes)
            matches_idx = self._match_to_anchors(true_boxes_t)
            classes_t = self._generate_class_t(matches_idx, true_boxes)
            coord_t = self._generate_coord_t(matches_idx, true_boxes)
        else:
            classes_t = self._generate_class_t_empty()
            coord_t = self._generate_coord_t_empty()

        label_t = np.concatenate((classes_t, coord_t), axis=1)

        if np.isnan(label_t).any():
            mask = np.any(np.isnan(label_t), axis=1)
            label_t[mask, :-4] = 0
            label_t[mask, 1] = 1
            label_t[mask, -4:] = self.anchors_t[mask]

        return label_t
