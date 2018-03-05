import numpy as np

from modelzoo.backend.tensor.metrics.Metric import Metric
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class MetricLocalization(Metric):
    @property
    def show(self):
        return self._show

    def show_result(self, img: Image):
        print('Localization Error:\n')
        if self._result is not None:
            for i in range(self._result.shape[0]):
                print('Box {4:d}:\n'
                      'X:{0:d} | Y:{1:d}\n'
                      'W:{2:d} | H:{3:d}\n'.format(self._result[i, 0], self._result[i, 1], self._result[i, 2],
                                                   self._result[i, 3], i))
        else:
            print("No matches")

    def __init__(self, iou_thresh=0.4, show=False):
        self._show = show
        self._result = None
        self.iou_thresh = iou_thresh

    def update(self, label_true: ImgLabel, label_pred: ImgLabel):
        boxes_true = BoundingBox.from_label(label_true)
        boxes_pred = BoundingBox.from_label(label_pred)

        losses = []
        for b_true in boxes_true:
            for b_pred in boxes_pred:
                if b_true.iou(b_pred) > self.iou_thresh:
                    loss = np.zeros((1, 4), dtype=int)
                    loss[0, 0] = int(abs(b_pred.cx - b_true.cx))
                    loss[0, 1] = int(abs(b_pred.cy - b_true.cy))
                    loss[0, 2] = int(abs(b_pred.w - b_true.w))
                    loss[0, 3] = int(abs(b_pred.h - b_true.h))
                    losses.append(loss)
                    break
        if len(losses) > 0:
            self._result = np.vstack(losses)
        else:
            self._result = None
        return self._result

    @property
    def result(self):
        return self._result
