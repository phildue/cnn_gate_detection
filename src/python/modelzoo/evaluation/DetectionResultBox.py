from modelzoo.evaluation.DetectionResult import DetectionResult
from utils.BoundingBox import BoundingBox


class DetectionResultBox:
    def __init__(self, true_positives: [BoundingBox], false_positives: [BoundingBox], false_negatives: [BoundingBox],
                 true_negatives: BoundingBox = None):
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self._result = DetectionResult(len(true_positives), len(false_positives), len(false_negatives),
                                       len(true_negatives))

    @property
    def recall(self):
        return self._result.recall

    @property
    def precision(self):
        return self._result.precision

    @property
    def fp_rate(self):
        return self._result.fp_rate

    def __repr__(self):
        return " _____________________\n" \
               "|Tp: {0:d} | Fp: {1:d}|\n" \
               "|Fn: {2:d} | Tn: {3:d}|\n" \
               " ---------------------".format(self.true_positives, self.false_positives, self.false_negatives,
                                               self.true_negatives)

    def __add__(self, other):
        return DetectionResult(self.true_positives + other.true_positives,
                               self.false_positives + other.false_positives,
                               self.false_negatives + other.false_negatives,
                               self.true_negatives + other.true_negatives)
