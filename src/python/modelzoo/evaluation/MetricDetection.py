from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.Metric import Metric
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show, LEGEND_TEXT
from utils.labels.ImgLabel import ImgLabel


class MetricDetection(Metric):
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
        self._result = None
        self.boxes_pred = None
        self.boxes_true = None
        self.false_positives = []
        self.true_positives = []
        self.false_negatives = []

    def evaluate(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.false_positives = []
        self.true_positives = []
        self.false_negatives = []

        self.boxes_pred = [b for b in BoundingBox.from_label(label_pred) #if
                           # (self.min_box_area < b.area < self.max_box_area and
                           #  self.min_aspect_ratio < b.h / b.w < self.max_aspect_ratio)
                           ]
        self.boxes_true = [b for b in BoundingBox.from_label(label_true) if
                           (self.min_box_area < b.area < self.max_box_area and
                            self.min_aspect_ratio < b.h / b.w < self.max_aspect_ratio)
                           ]

        for i,b in enumerate(self.boxes_true):
            box_matches = MetricDetection.match(b, self.boxes_pred, self.iou_thresh)
            if len(box_matches) is 0:
                self.false_negatives.append(i)
            else:
                new_matches = [m for m in box_matches if m not in self.true_positives]
                if len(new_matches) > 0:
                    self.true_positives.append(new_matches[0])
        self.false_positives = [m for m in range(len(self.boxes_pred)) if m not in self.true_positives]

        self.boxes_tp = [b for i, b in enumerate(self.boxes_pred) if i in self.true_positives]
        self.boxes_fp = [b for i, b in enumerate(self.boxes_pred) if i in self.false_positives]
        self.boxes_fn = [b for i, b in enumerate(self.boxes_true) if i in self.false_negatives]

        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        fp = len(self.boxes_pred) - len(self.true_positives)

        self._result = DetectionResult(tp, fp, fn)
        return self._result

    @staticmethod
    def match(box_true, boxes_pred, iou_thresh):
        matches = []
        for i, b in enumerate(boxes_pred):
            if box_true.iou(b) > iou_thresh and \
                    box_true.prediction == b.prediction:
                matches.append(i)

        return matches

    def update(self, label_true: ImgLabel, label_pred: ImgLabel):

        self.evaluate(label_true, label_pred)

        return self._result

    def show_result(self, img: Image):
        label_tp = BoundingBox.to_label(self.boxes_tp)
        label_fp = BoundingBox.to_label(self.boxes_fp)
        label_true = BoundingBox.to_label(self.boxes_true)
        print(self._result)
        # if self._result.true_positives < 0 or self._result.false_positives < 0 or self._result.false_negatives < 0:
        #     t = 0
        # else:
        #     t = 1
        t = 0
        show(img, 'result', labels=[label_true, label_fp, label_tp],
             colors=[COLOR_GREEN, COLOR_RED, (255, 255, 255)],
             legend=LEGEND_TEXT, t=t)

    @property
    def result(self):
        return self._result
