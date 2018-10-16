from modelzoo.evaluation.Metric import Metric

from modelzoo.evaluation.DetectionResult import DetectionResult
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show, LEGEND_TEXT
from utils.labels.ImgLabel import ImgLabel


class DetectionEvaluator(Metric):
    @property
    def show(self):
        return self._show

    def __init__(self, show_=False, iou_thresh=0.4, min_box_area=None, max_box_area=None, min_aspect_ratio=None,
                 max_aspect_ratio=None, store=False):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_box_area = max_box_area
        self.min_box_area = min_box_area
        self._store = store
        self._show = show_
        self.iou_thresh = iou_thresh
        self.result = None
        self.boxes_pred = None
        self.boxes_true = None
        self.false_positives_idx = []
        self.true_positives_idx = []
        self.false_negatives_idx = []
        self.boxes_tp = []
        self.boxes_fp = []
        self.boxes_fn = []

    def evaluate(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.false_positives_idx = []
        self.true_positives_idx = []
        self.false_negatives_idx = []

        self.boxes_pred = label_pred.objects
        self.boxes_true = [b for b in label_true.objects if
                           (self.min_box_area < b.poly.area < self.max_box_area and
                            self.min_aspect_ratio < b.poly.aspect_ratio < self.max_aspect_ratio)
                           ]

        for i, b in enumerate(self.boxes_true):
            matches_idx_pred = DetectionEvaluator.match(b, self.boxes_pred, self.iou_thresh)
            if len(matches_idx_pred) is 0:
                self.false_negatives_idx.append(i)
            else:
                new_matches = [m for m in matches_idx_pred if m not in self.true_positives_idx]
                if len(new_matches) > 0:
                    self.true_positives_idx.append(new_matches[0])
        self.false_positives_idx = [m for m in range(len(self.boxes_pred)) if m not in self.true_positives_idx]

        self.boxes_tp = [b for i, b in enumerate(self.boxes_pred) if i in self.true_positives_idx]
        self.boxes_fp = [b for i, b in enumerate(self.boxes_pred) if i in self.false_positives_idx]
        self.boxes_fn = [b for i, b in enumerate(self.boxes_true) if i in self.false_negatives_idx]

        self.result = DetectionResult(self.boxes_tp, self.boxes_fp, self.boxes_fn)
        return self.result

    @staticmethod
    def match(box_true, boxes_pred, iou_thresh):
        matches_idx_pred = []
        for i, b in enumerate(boxes_pred):
            if box_true.iou(b) > iou_thresh and \
                    box_true.prediction == b.prediction:
                matches_idx_pred.append(i)

        return matches_idx_pred


    def show_result(self, img: Image):
        label_tp = ImgLabel(self.boxes_tp)
        label_fp = ImgLabel(self.boxes_fp)
        label_true = ImgLabel(self.boxes_true)
        print(self.result)
        # if self._result.true_positives < 0 or self._result.false_positives < 0 or self._result.false_negatives < 0:
        #     t = 0
        # else:
        #     t = 1
        t = 0
        show(img, 'result', labels=[label_true, label_fp, label_tp],
             colors=[COLOR_GREEN, COLOR_RED, (255, 255, 255)],
             legend=LEGEND_TEXT, t=t)

