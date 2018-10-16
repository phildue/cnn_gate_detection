from modelzoo.visuals.plots.BasePlot import BasePlot


class PlotApAngle(BasePlot):
    def __init__(self, precision, angle, size=(6, 5), font_size=12, title='', line_style='--'):
        super().__init__(angle, precision, size, font_size, title, line_style, x_label='Angle[rad]',
                         y_label='Precision')
