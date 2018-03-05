import numpy as np

from modelzoo.backend.tensor.metrics.Metric import Metric
from modelzoo.evaluation.DetectionResult import DetectionResult
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, COLOR_BLUE, LEGEND_BOX, save_labeled, show
from utils.labels.ImgLabel import ImgLabel


class MetricOneGate(Metric):
    idx = 0

    @property
    def show(self):
        return self._show

    def __init__(self, show_=False, iou_thresh=0.4, store_path=None):
        self._store = store_path
        self._show = show_
        self.iou_thresh = iou_thresh
        self._result = None
        self._boxes_highest_conf = None
        self._boxes_pred = None
        self._boxes_true = None

    def evaluate(self, label_true: ImgLabel, label_pred: ImgLabel):
        self._boxes_pred = BoundingBox.from_label(label_pred)
        self._boxes_true = BoundingBox.from_label(label_true)
        self._boxes_highest_conf = []

        tp, tn, fp, fn = 0, 0, 0, 0
        loss = None

        b_pred = None
        highest_conf = 0
        for b in self._boxes_pred:
            if b.c > highest_conf:
                highest_conf = b.c
                b_pred = b

        if b_pred is None:
            if not self._boxes_true:
                tn = 1
            else:
                fn = 1
        else:
            self._boxes_highest_conf = [b_pred]
            if not self._boxes_true:
                fp = 1
            else:
                b_true = self._boxes_true[0]
                if b_true.iou(b_pred) >= self.iou_thresh and \
                                np.argmax(b_true.probs) == np.argmax(b_pred.probs):
                    tp = 1
                    self._boxes_highest_conf.append(b_pred)
                else:
                    fn = 1
                loss = np.zeros((1, 4), dtype=int)
                loss[0, 0] = int(abs(b_pred.cx - b_true.cx))
                loss[0, 1] = int(abs(b_pred.cy - b_true.cy))
                loss[0, 2] = int(abs(b_pred.w - b_true.w))
                loss[0, 3] = int(abs(b_pred.h - b_true.h))

        self._result = DetectionResult(tp, fp, fn, tn), loss

    def update(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.evaluate(label_true, label_pred)

        return self._result

    def show_result(self, img: Image):
        label_highest_conf = BoundingBox.to_label(self._boxes_highest_conf)
        label_pred = BoundingBox.to_label(self._boxes_pred)
        label_true = BoundingBox.to_label(self._boxes_true)
        print(self._result[0])
        if self._result[1] is not None:
            loss = self._result[1]
            print('Localization Error:\n'
                  'X:{0:d} | Y:{1:d}\n'
                  'W:{2:d} | H:{3:d}\n'.format(loss[0, 0], loss[0, 1], loss[0, 2],
                                               loss[0, 3]))

        show(img.bgr, 'result', labels=[label_true, label_pred, label_highest_conf],
             colors=[COLOR_GREEN, COLOR_RED, COLOR_BLUE],
             legend=LEGEND_BOX, t=1)
        if self._store is not None:
            save_labeled(img.bgr, self._store + 'result{0:04d}'.format(self.idx),
                         labels=[label_true, label_pred, label_highest_conf],
                         colors=[COLOR_GREEN, COLOR_RED, COLOR_BLUE],
                         legend=LEGEND_BOX, )
            self.idx += 1

    @property
    def result(self):
        return self._result
