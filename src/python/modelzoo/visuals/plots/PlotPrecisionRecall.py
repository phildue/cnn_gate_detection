import matplotlib.pyplot as plt
import numpy as np

from modelzoo.visuals.plots.BasePlot import BasePlot


class PlotPrecisionRecall(BasePlot):
    def __init__(self, precision, recall, size=(6, 5), font_size=12, title='Precision-Recall', line_style='--'):
        super().__init__(x_data=recall, y_data=precision, size=size, font_size=font_size, title=title,
                         line_style=line_style, y_label='Precision', x_label='Recall')
        self.ap = np.mean(precision)

    def create_fig(self):
        plt.annotate('meanAP= ' + str(np.round(self.ap, 2)), xy=(0, 0), xytext=(0, 0), fontsize=self.font_size)
        super().create_fig()
