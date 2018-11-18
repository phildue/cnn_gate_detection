import numpy as np

from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
from visuals import BaseMultiPlot

cd_work()
n_epochs = 30
models = ['out/0108/multiscale2208x208']

epochs = []
val_mAPs = []
mAP = []
losses = []
val_loss = []
for m in models:
    log_csv = load_file(m + '/log.csv')
    mat_csv = np.zeros((n_epochs, 6), dtype=np.float)
    for i in range(1, n_epochs + 1):
        mat_csv[i - 1, 0] = float(log_csv[i][0])
        mat_csv[i - 1, 1] = float(log_csv[i][1])
        mat_csv[i - 1, 2] = float(log_csv[i][2])
        mat_csv[i - 1, 3] = float(log_csv[i][3])
        mat_csv[i - 1, 4] = float(log_csv[i][4])
        mat_csv[i - 1, 5] = float(log_csv[i][5])

    epochs.append(mat_csv[:n_epochs, 0])
    mAP.append(mat_csv[:n_epochs, 1])
    val_mAPs.append(mat_csv[:n_epochs, 4])
    losses.append(mat_csv[:, 3])
    lr = mat_csv[:, 2]
    val_loss.append(mat_csv[:, 5])

training_plot = BaseMultiPlot(x_data=[epochs] * 2, x_label='Epochs',
                              y_data=[losses, val_loss],
                              legend=['mAP', 'loss/10', 'val_loss/10', 'val_Map'],
                              # y_lim=(0.0, 1.0),
                              title='Training')

training_plot.show()
