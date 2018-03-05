import numpy as np

from src.python.modelzoo.evaluation import DetectionResult
from src.python.modelzoo.evaluation import EvaluatorPrecisionRecall
from src.python.utils.labels import GateLabel


def group_and_plot(measurements: [({float: DetectionResult}, GateLabel)], output_path='./', n_bins=10, eucl_min=5,
                   eucl_max=30,
                   ang_max=np.pi / 2,
                   ang_min=-np.pi / 2, block=True, fig_size=(6, 5), fontsize=12):
    group_eucl, group_pitch, group_yaw, group_roll, group_forw, group_side, group_lift = group_by_pos(measurements)

    plot_ap_bin(group_eucl, 'Euclidian Distance', 5, 30, 'n = ' + str(len(measurements)), output_path, n_bins, fontsize,
                fig_size, False)
    plot_ap_bin(group_pitch, 'Pitch angle', ang_min, ang_max, 'n = ' + str(len(measurements)), output_path, n_bins,
                fontsize,
                fig_size, False)
    plot_ap_bin(group_yaw, 'Yaw Angle', ang_min, ang_max, 'n = ' + str(len(measurements)), output_path, n_bins,
                fontsize,
                fig_size, False)
    plot_ap_bin(group_roll, 'Roll Angle', ang_min, ang_max, 'n = ' + str(len(measurements)), output_path, n_bins,
                fontsize,
                fig_size, False)
    plot_ap_bin(group_forw, 'Distance Front', 5, 30, 'n = ' + str(len(measurements)), output_path, n_bins, fontsize,
                fig_size, False)
    plot_ap_bin(group_side, 'Distance Side', 5, 30, 'n = ' + str(len(measurements)), output_path, n_bins, fontsize,
                fig_size, False)
    plot_ap_bin(group_lift, 'Lift', 5, 30, 'n = ' + str(len(measurements)), output_path, n_bins, fontsize,
                fig_size, block)


def plot_ap_bin(group, xlabel, bin_min, bin_max, title='', output_path=None, n_bins=10, fontsize=12, fig_size=(5, 4),
                block=True):
    sorted_eucl = sort_results(group, bin_min, bin_max, n_bins)
    means, bins = mean_groups(sorted_eucl)
    plot_step(y=np.array(means), x=np.array(bins), xlabel=xlabel, ylabel='meanAP',
              output_file=output_path + 'meanAP-' + xlabel + '.png', block=block, title=title,
              size=fig_size, fontsize=fontsize)


def group_by_pos(measurements):
    group_eucl = []
    group_pitch = []
    group_yaw = []
    group_roll = []
    group_forw = []
    group_side = []
    group_lift = []
    for m in measurements:
        if m[1] is None: continue
        position = m[1].pose
        group_eucl.append((m[0], np.sqrt(position.dist_forward ** 2 + position.lift ** 2 + position.dist_side ** 2)))
        group_pitch.append((m[0], position.pitch))
        group_roll.append((m[0], position.roll))
        group_yaw.append((m[0], position.yaw))
        group_forw.append((m[0], position.dist_forward))
        group_side.append((m[0], position.dist_side))
        group_lift.append((m[0], position.lift))

    return group_eucl, group_pitch, group_yaw, group_roll, group_forw, group_side, group_lift


def sort_results(results, bin_min=0.0, bin_max=1.0, n_bins=10):
    sorted_results = {}
    for i in np.linspace(bin_min, bin_max, n_bins):
        sorted_results[i] = []

    for e in results:
        result = e[0]
        key = e[1]
        for i in range(1, len(sorted_results)):
            bins = sorted(list(sorted_results.keys()))
            if bins[i - 1] < key <= bins[i]:
                sorted_results[bins[i]].append(result)
    return sorted_results


def mean_groups(group):
    means = []
    bins = sorted(list(group.keys()))
    for bin in bins:
        bin_total = {}
        confidences = np.linspace(1.0, 0, 11)
        for c in confidences:
            bin_total[c] = DetectionResult(0, 0, 0)
        for r in group[bin]:
            if r is not None:
                for c in confidences:
                    bin_total[c].false_positives += r[c].false_positives
                    bin_total[c].true_positives += r[c].true_positives
                    bin_total[c].false_negatives += r[c].false_negatives
        print("AP-Position::Number of members in group " + str(bin) + ": " + str(len(group[bin])))

        values = [bin_total[k] for k in reversed(sorted(bin_total.keys()))]
        means.append(EvaluatorPrecisionRecall.interp(values)[1])
    return means, bins
