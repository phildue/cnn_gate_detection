import matplotlib.pyplot as plt

from modelzoo.visuals.plots import PyPlot


class PlotCollection(PyPlot):
    def __init__(self, subplots: [PyPlot], grid: (int, int), size=(6, 5), title=''):
        self.grid = grid
        self.subplots = subplots
        self.title = title
        self.size = size

    def show(self, block=True):
        plt.figure(self.size)
        self.create_fig()
        plt.show(block=block)

    def save(self, filename: str = None):
        filename = './' + self.title + '.png' if filename is None else filename
        plt.figure(self.size)
        self.create_fig()
        plt.savefig(filename)

    def create_fig(self):
        for i, p in enumerate(self.subplots):
            plt.subplot(self.grid[0], self.grid[1], i)
            p.create_fig()
