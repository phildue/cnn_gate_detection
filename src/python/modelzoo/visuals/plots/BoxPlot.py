# coding=utf-8
import matplotlib.pyplot as plt

from modelzoo.visuals.plots.BasePlot import BasePlot


class BoxPlot(BasePlot):
    def __init__(self, x_data, size=(6, 5), font_size=12, title='', line_style='--', x_label='x', y_label='y'):
        super().__init__(x_data, None, size, font_size, title, line_style, x_label, y_label)

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        plt.boxplot(self.x_data)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
