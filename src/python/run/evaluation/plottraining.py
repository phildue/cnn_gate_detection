import argparse

import numpy as np

from modelzoo.evaluation.utils import average_precision_recall
from modelzoo.visuals.plots.BaseMultiPlot import BaseMultiPlot
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work


def plot_training(work_dir, n_epochs):
    log_csv = load_file(work_dir + 'log.csv')
    mat_csv = np.zeros((n_epochs, 4), dtype=np.float)
    for i in range(1, n_epochs + 1):
        mat_csv[i - 1, 0] = float(log_csv[i][0])
        mat_csv[i - 1, 1] = float(log_csv[i][1])
        mat_csv[i - 1, 2] = float(log_csv[i][2])
        mat_csv[i - 1, 3] = float(log_csv[i][3])

    epochs = mat_csv[:, 0]
    loss = mat_csv[:, 1]
    lr = mat_csv[:, 2]
    val_loss = mat_csv[:, 3]
    n = len(epochs)
    mAP = np.zeros((n,))
    for i in range(n):
        file = load_file(work_dir + 'results/set_2--{0:03d}.pkl'.format(i))
        results = [ResultByConfidence(d) for d in file['results']['MetricDetection']]
        avg_precision, recall = average_precision_recall(results)
        mAP[i] = np.mean(avg_precision)

    training_plot = BaseMultiPlot(x_data=[epochs, epochs, epochs, epochs], x_label='Epochs',
                                  y_data=[mAP, loss / 10, lr * 100, val_loss / 10],
                                  legend=['mAP', 'loss/10', 'lr*100', 'val_loss/10'],
                                  y_lim=(0.0, 1.0),
                                  title='Training' + work_dir)
    return training_plot


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Model Directory",
                    type=str)
parser.add_argument("epoch", help="Epoch until history will be printed", type=int)
args = parser.parse_args()

cd_work()

plot_training(args.dir, args.epoch).show(True)
