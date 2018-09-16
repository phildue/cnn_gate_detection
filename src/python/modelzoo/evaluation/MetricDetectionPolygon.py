import numpy as np
from shapely.geometry import Polygon

from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.Metric import Metric
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, LEGEND_BOX, show
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class BoundingPolygon:
    def __init__(self, p: Polygon, class_probs):
        self.class_probs = class_probs
        self.p = p

    @property
    def area(self):
        return self.p.area

    def iou(self, other):
        return self.p.intersection(other.p).area / self.p.union(other.p).area

    @property
    def prediction(self):
        return np.argmax(self.class_probs)


class MetricDetectionPolygon(Metric):
    @property
    def show(self):
        return self._show

    def __init__(self, show_=False, iou_thresh=0.4, min_box_area=None, max_box_area=None, store=False):
        self.max_box_area = max_box_area
        self.min_box_area = min_box_area
        self._store = store
        self._show = show_
        self.iou_thresh = iou_thresh
        self._result = None
        self._boxes_correct = None
        self._boxes_pred = None
        self._boxes_true = None

    def to_bounding_polygon(self, obj: ObjectLabel):
        obj_tuple = [obj.gate_corners.bottom_left, obj.gate_corners.bottom_right, obj.gate_corners.top_right,
                     obj.gate_corners.top_left]
        class_probs = np.zeros((len(ObjectLabel.classes) + 1,))
        class_probs[ObjectLabel.name_to_id('gate')] = obj.confidence
        return BoundingPolygon(Polygon(obj_tuple), class_probs)

    def evaluate(self, label_true: ImgLabel, label_pred: ImgLabel):
        self.label_true = label_true
        self.label_pred = label_pred
        polygons_pred = [self.to_bounding_polygon(obj) for obj in label_pred.objects]
        polygons_true = [self.to_bounding_polygon(obj) for obj in label_true.objects]
        self._boxes_pred = [b for b in polygons_pred if
                            self.min_box_area < b.area < self.max_box_area]
        self._boxes_true = [b for b in polygons_true if
                            self.min_box_area < b.area < self.max_box_area]
        self._boxes_correct = []
        true_positives = []

        false_negatives = []
        for b in self._boxes_true:
            box_matches = MetricDetectionPolygon.match(b, self._boxes_pred, self.iou_thresh)
            if len(box_matches) is 0:
                false_negatives.append(b)
            else:
                new_matches = [m for m in box_matches if m not in true_positives]
                if len(new_matches) > 0:
                    true_positives.append(new_matches[0])
                    self._boxes_correct.append(self._boxes_pred[new_matches[0]])

        tp = len(true_positives)
        fn = len(false_negatives)
        fp = len(self._boxes_pred) - len(true_positives)

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
        # label_correct = BoundingBox.to_label(self._boxes_correct)
        print(self._result)
        # if self._result.true_positives < 0 or self._result.false_positives < 0 or self._result.false_negatives < 0:
        #     t = 0
        # else:
        #     t = 1
        t = 0
        show(img.bgr, 'result', labels=[self.label_true, self.label_pred],
             colors=[COLOR_GREEN, COLOR_RED, (255, 255, 255)],
             legend=LEGEND_BOX, t=t)

    @property
    def result(self):
        return self._result
