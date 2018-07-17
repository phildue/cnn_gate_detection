class DetectionResult:
    def __init__(self, true_positives: int, false_positives: int, false_negatives: int, true_negatives: int = 0):
        if true_positives < 0 or false_positives < 0 or false_negatives < 0:
            print('Warning')
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.true_positives = true_positives
        self.true_negatives = true_negatives

    @property
    def recall(self):
        try:
            return self.true_positives / (self.false_negatives + self.true_positives)
        except ZeroDivisionError:
            return 0.0

    @property
    def precision(self):
        try:
            return self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return 0.0

    @property
    def fp_rate(self):
        try:
            return self.false_positives / (self.true_negatives + self.false_positives)
        except ZeroDivisionError:
            return 0.0

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
