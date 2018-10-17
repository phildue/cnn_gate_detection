from modelzoo.visuals.plots.BasePlot import BasePlot


class PlotPrecisionPosition(BasePlot):
    def __init__(self, precision, position, size=(6, 5), font_size=12, title='', line_style='--'):
        super().__init__(position, precision, size=size, font_size=font_size, title=title, line_style=line_style,
                         x_label='Position[m]',
                         y_label='Precision', )
