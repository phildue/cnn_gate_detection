import numpy as np
from labels.GateLabel import GateLabel

from src.python.modelzoo.backend.visuals import BasePlot
from src.python.modelzoo.evaluation import EvaluatorPrecisionRecall
from src.python.modelzoo.evaluation import ResultByConfidence
from src.python.utils.labels.Pose import Pose


class PositionPlotCreator:
    def __init__(self, results_label: [(ResultByConfidence, GateLabel)]):
        self.precisions = [np.mean(EvaluatorPrecisionRecall.interp(r[0])) for r in results_label]
        self.positions = [r[1].pose for r in results_label]

    @staticmethod
    def __eucl_dist(position: Pose):
        return np.sqrt(position.dist_forward ** 2 + position.lift ** 2 + position.dist_side ** 2)

    @staticmethod
    def __extract_metric(positions: [Pose], key: str):
        positions = [p for p in positions if p is not None]
        if key is 'pitch':
            return [p.pitch for p in positions]
        elif key is 'yaw':
            return [p.yaw for p in positions]
        elif key is 'roll':
            return [p.roll for p in positions]
        elif key is 'forward':
            return [p.forw for p in positions]
        elif key is 'side':
            return [p.side for p in positions]
        elif key is 'lift':
            return [p.lift for p in positions]
        else:
            return [PositionPlotCreator.__eucl_dist(p) for p in positions]

    @staticmethod
    def __sort(results, position):

        mat = np.array([results, position])
        mat = mat[::, mat[1,].argsort()]

        return mat[0, :], mat[1, :]

    @staticmethod
    def __sort_by_metric(positions, precisions, metric):
        positions_extracted = PositionPlotCreator.__extract_metric(positions, metric)

        return PositionPlotCreator.__sort(precisions, positions_extracted)

    def create(self, metric: str, x_label='', line_style='--',
               title='', size=(6, 5), font_size=12):

        precisions, positions = self.__sort_by_metric(self.positions, self.precisions, metric)

        return BasePlot(x_data=positions, y_data=precisions, size=size, font_size=font_size, title=title,
                        line_style=line_style,
                        x_label=x_label, y_label='Precision')

    def create_bin(self, metric: str, x_label='', line_style='--',
                   title='', size=(6, 5), font_size=12, bin_size=10):

        precisions, positions = self.__sort_by_metric(self.positions, self.precisions, metric)
        precisions, positions = self.__group_in_bins(precisions, positions, bin_size)

        return BasePlot(x_data=positions, y_data=precisions, size=size, font_size=font_size, title=title,
                        line_style=line_style,
                        x_label=x_label, y_label='Precision')

    @staticmethod
    def __group_in_bins(precisions, positions, bin_size=10):
        n_steps = int(np.floor(len(precisions) / bin_size)) - 1
        bin_key = np.linspace(np.min(positions), np.max(positions), n_steps)
        bin_content = [np.mean(precisions[i * bin_size:(i + 1) * bin_size]) for i in range(0, n_steps)]

        return bin_content, bin_key
