import numpy as np

from modelzoo.backend.visuals.plots.Heatmap import Heatmap
from utils.fileaccess.LabelFileParser import LabelFileParser


class SetAnalyzer:
    def __init__(self, img_shape, path):
        self.img_shape = img_shape
        self.label_reader = LabelFileParser(path, 'xml')

    def get_heat_map(self):

        labels = self.label_reader.read()
        y_limit = self.img_shape[0]
        label_sum = np.zeros(self.img_shape)
        for l in labels:
            label_map = np.zeros(self.img_shape)
            for o in l.objects:
                y_max = y_limit - o.y_min
                y_min = y_limit - o.y_max
                label_map[int(y_min):int(y_max), int(o.x_min): int(o.x_max)] = 1
            label_sum += label_map

        label_scaled = label_sum / np.max(label_sum)

        return Heatmap(label_scaled)
