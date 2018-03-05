import matplotlib.pyplot as plt
import numpy as np

from modelzoo.backend.visuals.plots.BasePlot import BasePlot


class PlotPrecisionRecallMulti(BasePlot):
    def __init__(self, precision, recall, size=(6, 5), font_size=12, title='Precision-Recall', line_style=None,
                 legend=None):
        if line_style is None:
            line_style = ['r--', 'g--']
        super().__init__(x_data=recall, y_data=precision, size=size, font_size=font_size, title=title,
                         line_style=line_style, y_label='Precision', x_label='Recall')

        self.legend = legend
        self.ap = [0] * len(precision)
        for i, p in enumerate(precision):
            self.ap[i] = np.mean(p)

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.grid(True, axis='both', which='both')
        hs = []
        for i, p in enumerate(self.y_data):
            plt.annotate('meanAP= ' + str(np.round(self.ap[i], 2)), xy=(0, i / 10), xytext=(0, i / 10),
                         fontsize=self.font_size)
            hs += plt.plot(self.x_data[i], p, self.line_style[i])
        if self.legend is not None:
            plt.legend(hs, self.legend)
