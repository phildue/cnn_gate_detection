import numpy as np

from modelzoo.evaluation.DetectionResult import DetectionResult


class ResultByConfidence:
    def __init__(self, results: {float: DetectionResult}):
        self.results = results

    @property
    def values(self):
        return [self.results[k] for k in reversed(sorted(self.results.keys()))]

    def __add__(self, other):
        confidences = list(reversed(sorted(self.results.keys())))

        total = {}
        for c in confidences:
            total[c] = self.results[c] + other.results[c]

        return ResultByConfidence(total)


