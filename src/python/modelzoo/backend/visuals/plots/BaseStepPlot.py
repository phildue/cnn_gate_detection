import matplotlib.pyplot as plt

from modelzoo.backend.visuals.plots.BasePlot import BasePlot


class BaseStepPlot(BasePlot):
    def __init__(self, x_data, y_data, size=(6, 5), font_size=12, title='', line_style='--', x_label='x', y_label='y'):
        super().__init__(x_data, y_data, size, font_size, title, line_style, x_label, y_label)

    def create_fig(self):
        plt.step(self.x_data, self.y_data)
        super().create_fig()
