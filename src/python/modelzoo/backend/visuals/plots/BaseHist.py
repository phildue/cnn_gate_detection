import matplotlib.pyplot as plt

from modelzoo.backend.visuals.plots.BasePlot import BasePlot


class BaseHist(BasePlot):
    def __init__(self, y_data, size=(6, 5), font_size=12, title='', line_style='--', x_label='x', y_label='y',
                 n_bins=1):
        super().__init__(None, y_data, size, font_size, title, line_style, x_label, y_label)
        self.bin_width = n_bins

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        plt.hist(self.y_data, self.bin_width, facecolor='green', alpha=0.75)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
