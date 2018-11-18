import matplotlib.pyplot as plt

from visuals import PyPlot


class Heatmap(PyPlot):
    def show(self, block=True):
        plt.figure()
        plt.imshow(self.create_fig())
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.colorbar()
        plt.show(block)

    def save(self, file_path):
        plt.imsave(file_path, self.create_fig())

    def create_fig(self):
        return self.mat

    def __init__(self, mat, x_label, y_label, title):
        self.title = title
        self.y_label = y_label
        self.x_label = x_label
        self.mat = mat

