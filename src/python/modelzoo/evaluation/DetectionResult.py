import numpy as np

from utils.labels.ObjectLabel import ObjectLabel


class DetectionResult:
    def __init__(self, true_positives: [ObjectLabel], false_positives: [ObjectLabel], false_negatives: [ObjectLabel],
                 true_negatives: [ObjectLabel] = None):

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives

        if len(true_positives) < 0 or len(false_positives) < 0 or len(false_negatives) < 0:
            print('Warning')

        self.n_fn = len(false_negatives)
        self.n_fp = len(false_positives)
        self.n_tp = len(true_positives)
        self.n_tn = len(true_negatives) if true_negatives is not None else 0

    @property
    def recall(self):
        try:
            return self.n_tp / (self.n_fn + self.n_tp)
        except ZeroDivisionError:
            return 0.0

    @property
    def precision(self):
        try:
            return self.n_tp / (self.n_tp + self.n_fp)
        except ZeroDivisionError:
            return 0.0

    @property
    def fp_rate(self):
        try:
            return self.n_fp / (self.n_tn + self.n_fp)
        except ZeroDivisionError:
            return 0.0

    @property
    def precision_conf(self):
        confidences = np.linspace(0, 1.0, 11)
        precision = np.zeros_like(confidences)
        for j, c in enumerate(confidences):
            n_fp = len([o for o in self.false_positives if o.confidence > c])
            n_tp = len([o for o in self.true_positives if o.confidence > c])
            if np.any(np.array([n_fp, n_tp]) < 0):
                raise ValueError("Weird Numbers")
            precision[j] = (n_tp / (n_fp + n_tp))
        return precision

    @property
    def recall_conf(self):
        confidences = np.linspace(0, 1.0, 11)
        recall = np.zeros_like(confidences)
        for j, c in enumerate(confidences):
            n_tp = len([o for o in self.true_positives if o.confidence > c])
            n_fn = len(self.true_positives + self.false_negatives) - n_tp
            if np.any(np.array([n_tp, n_fn]) < 0):
                raise ValueError("Weird Numbers")
            recall[j] = (n_tp / (n_fn + n_tp))
        return recall

    def __repr__(self):
        return " _____________________\n" \
               "|Tp: {0:d} | Fp: {1:d}|\n" \
               "|Fn: {2:d} | Tn: {3:d}|\n" \
               " ---------------------".format(self.n_tp, self.n_fp, self.n_fn,
                                               self.n_tn)

    def __add__(self, other):
        return DetectionResult(self.true_positives + other.true_positives,
                               self.false_positives + other.false_positives,
                               self.false_negatives + other.false_negatives,
                               self.true_negatives + other.true_negatives)
