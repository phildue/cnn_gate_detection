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

    def __init__(self, show_=False, iou_thresh=0.4, store=False):
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

        for b_true in self._boxes_true:
            match = False
            for b_pred in self._boxes_pred:
                if b_true.iou(b_pred) >= self.iou_thresh and \
                                np.argmax(b_true.probs) == np.argmax(b_pred.probs):
                    tp += 1
                    self._boxes_correct.append(b_pred)
                    match = True
                    break
            if not match:
                fn += 1

        fp = len(self._boxes_pred) - tp

        self._result = DetectionResult(tp, fp, fn)

    def update(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.evaluate(label_true, label_pred)

        return self._result

    def show_result(self, img: Image):
        label_correct = BoundingBox.to_label(self._boxes_correct)
        label_pred = BoundingBox.to_label(self._boxes_pred)
        label_true = BoundingBox.to_label(self._boxes_true)
        print(self._result)
        show(img.bgr, 'result', labels=[label_true, label_pred, label_correct],
             colors=[COLOR_GREEN, COLOR_RED, (255, 255, 255)],
             legend=LEGEND_BOX)

    @property
    def result(self):
        return self._result
