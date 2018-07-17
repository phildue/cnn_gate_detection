import numpy as np

from modelzoo.evaluation.DetectionResult import DetectionResult


class ResultByConfidence:
    def __init__(self, results: {float: DetectionResult}):
        self.results = results

    @property
    def values(self):
        return [self.results[k] for k in reversed(sorted(self.results.keys()))]

    @property
    def confidences(self):
        return list(reversed(sorted(self.results.keys())))

    @property
    def precisions(self):
        mat = np.zeros((len(self.results.keys(), )))

        for i, c in enumerate(list(reversed(sorted(self.results.keys())))):
            mat[i] = self.results[c].precision

        return mat

    @property
    def recalls(self):
        mat = np.zeros((len(self.results.keys(), )))

        for i, c in enumerate(list(reversed(sorted(self.results.keys())))):
            mat[i] = self.results[c].recall

        return mat

    @property
    def true_positives(self):
        mat = np.zeros((len(self.results.keys(), )))

        for i, c in enumerate(list(reversed(sorted(self.results.keys())))):
            mat[i] = self.results[c].true_positives

        return mat

    @property
    def false_positives(self):
        mat = np.zeros((len(self.results.keys(), )))

        for i, c in enumerate(list(reversed(sorted(self.results.keys())))):
            mat[i] = self.results[c].false_positives

        return mat

    @property
    def false_negatives(self):
        mat = np.zeros((len(self.results.keys(), )))

        for i, c in enumerate(list(reversed(sorted(self.results.keys())))):
            mat[i] = self.results[c].false_negatives

        return mat

    @property
    def average_precision(self):
        avg_precision = 0
        for v in self.values:
            avg_precision += 1 / len(self.confidences) * v.precision
        return avg_precision

    def __add__(self, other):
        confidences = list(reversed(sorted(self.results.keys())))

        total = {}
        for c in confidences:
            if (other.results[c].false_positives < 0 or
                    other.results[c].true_positives < 0 or
                    other.results[c].false_negatives < 0):

                print('Warning weird numbers')
                return self
            else:
                total[c] = self.results[c] + other.results[c]

        return ResultByConfidence(total)
