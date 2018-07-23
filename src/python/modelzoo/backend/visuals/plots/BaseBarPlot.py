import matplotlib.pyplot as plt
import numpy as np
from modelzoo.backend.visuals.plots.BasePlot import BasePlot


class BaseBarPlot(BasePlot):
    def __init__(self, x_data, y_data, size=(6, 5), font_size=12, title='', line_style='--', x_label='x', y_label='y',
                 width=0.1,
                 colors='blue', legend=None, names=None):
        super().__init__(x_data, y_data, size, font_size, title, line_style, x_label, y_label, )
        self.names = names
        self.legend = legend
        self.colors = colors
        self.width = width

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        h = []
        if isinstance(self.y_data, list):
            for i in range(len(self.y_data)):
                h += plt.bar(np.array(self.x_data[i]) - (len(self.y_data) / 2) * self.width + i * self.width,
                             self.y_data[i], color=self.colors[i],
                             align='center', alpha=0.5,
                             width=self.width)

        else:
            h += plt.bar(self.x_data, self.y_data, color=self.colors, align='center', alpha=0.5, width=self.width)
        plt.grid(b=True, which='both', color='0.65', linestyle='-')
        if self.names is not None:
            plt.xticks(self.x_data[0], self.names)
        if self.legend is not None:
            plt.legend(self.legend)
