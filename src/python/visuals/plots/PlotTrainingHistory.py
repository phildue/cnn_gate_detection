import matplotlib.pyplot as plt

from visuals import BasePlot


class PlotTrainingHistory(BasePlot):
    def __init__(self, training_history, metrics=['loss', 'val_loss']):
        super().__init__(x_data=None, y_data=None, title='Training History', y_label='Metric', x_label='Epoch')
        self.metrics = metrics
        self.training_history = training_history

    def create_fig(self):
        plt.title(self.title)
        plt.xlabel(self.x_label, fontsize=self.font_size)
        plt.ylabel(self.y_label, fontsize=self.font_size)
        for m in self.metrics:
            plt.plot(self.training_history[m])
        plt.legend(self.metrics, loc='upper left')
