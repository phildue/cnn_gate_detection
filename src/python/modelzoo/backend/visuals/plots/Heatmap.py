import matplotlib.pyplot as plt

from modelzoo.backend.visuals.plots.PyPlot import PyPlot


class Heatmap(PyPlot):
    def show(self, block=True):
        plt.imshow(self.create_fig())
        plt.show()

    def save(self, file_path):
        plt.imsave(file_path, self.create_fig())

    def create_fig(self):
        return plt.cm.jet(self.mat)

    def __init__(self, mat):
        self.mat = mat
