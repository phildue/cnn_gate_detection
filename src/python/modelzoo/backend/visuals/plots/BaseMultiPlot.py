import matplotlib.pyplot as plt
import numpy as np

from modelzoo.backend.visuals.plots.BasePlot import BasePlot


class BaseMultiPlot(BasePlot):
    def __init__(self, x_data, y_data, size=(6, 5), font_size=12, title='', line_style=None, x_label='x', y_label='y',
                 legend=None, y_lim=None):
        if line_style is None:
            line_style = '--' * len(y_data)
        super().__init__(x_data, y_data, size, font_size, title, line_style, x_label, y_label)
        if y_lim is None:
            y_lim = np.min(self.y_data) - 0.1 * np.min(self.y_data), np.max(self.y_data) + 0.1 * np.max(self.y_data)

        self.y_lim = y_lim
        self.legend = legend

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        hs = []
        for i in range(len(self.y_data)):
            hs += plt.plot(self.x_data[i], self.y_data[i], self.line_style[i])
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.ylim(self.y_lim)
        plt.grid(b=True, which='both', color='0.65', linestyle='-')
        if self.legend is not None:
            plt.legend(hs, self.legend)
