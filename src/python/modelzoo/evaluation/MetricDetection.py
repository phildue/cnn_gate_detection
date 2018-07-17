import numpy as np

from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.Metric import Metric
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, LEGEND_BOX, show
from utils.labels.ImgLabel import ImgLabel


class MetricDetection(Metric):
    @property
    def show(self):
        return self._show

    def __init__(self, show_=False, iou_thresh=0.4, min_box_area=None, store=False):
        self.box_area = min_box_area
        self._store = store
        self._show = show_
        self.iou_thresh = iou_thresh
        self._result = None
        self._boxes_correct = None
        self._boxes_pred = None
        self._boxes_true = None

    def evaluate(self, label_true: ImgLabel, label_pred: ImgLabel):
        self._boxes_pred = BoundingBox.from_label(label_pred)
        self._boxes_true = BoundingBox.from_label(label_true)

        self._boxes_correct = []
        tp = 0
        fn = 0
        n_too_small = 0

        for b_true in self._boxes_true:
            match = False

            for b_pred in self._boxes_pred:
                if b_true.iou(b_pred) >= self.iou_thresh and \
                        np.argmax(b_true.probs) == np.argmax(b_pred.probs):
                    if b_true.area < self.box_area:
                        n_too_small += 1
                    else:
                        tp += 1
                        self._boxes_correct.append(b_pred)
                        match = True
                    break
            if not match and b_true.area > self.box_area:
                fn += 1

        fp = len(self._boxes_pred) - n_too_small - tp

        self._result = DetectionResult(tp, fp, fn)
        return self._result

    def update(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.evaluate(label_true, label_pred)

        return self._result

    def show_result(self, img: Image):
        label_correct = BoundingBox.to_label(self._boxes_correct)
        label_pred = BoundingBox.to_label(self._boxes_pred)
        label_true = BoundingBox.to_label(self._boxes_true)
        print(self._result)
        # if self._result.true_positives < 0 or self._result.false_positives < 0 or self._result.false_negatives < 0:
        #     t = 0
        # else:
        #     t = 1
        t = 0
        show(img.bgr, 'result', labels=[label_true, label_pred, label_correct],
             colors=[COLOR_GREEN, COLOR_RED, (255, 255, 255)],
             legend=LEGEND_BOX, t=t)

    @property
    def result(self):
        return self._result
