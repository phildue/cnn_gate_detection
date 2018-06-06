import matplotlib.pyplot as plt

from modelzoo.backend.visuals.plots.PyPlot import PyPlot


class BasePlot(PyPlot):
    def __init__(self, x_data, y_data, size=(6, 5), font_size=12, title='', line_style='--', x_label='x', y_label='y'):
        self.y_data = y_data
        self.x_data = x_data
        self.y_label = y_label
        self.x_label = x_label
        self.line_style = line_style
        self.title = title
        self.size = size
        self.font_size = font_size

    def show(self, block=True):
        plt.figure(figsize=self.size)
        self.create_fig()
        plt.show(block=block)

    def save(self, filename: str = None, transparent=False):
        filename = './' + self.title + '.png' if filename is None else filename
        plt.figure(figsize=self.size)
        self.create_fig()
        plt.savefig(filename, transparent=transparent)

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        plt.plot(self.x_data, self.y_data, self.line_style)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
