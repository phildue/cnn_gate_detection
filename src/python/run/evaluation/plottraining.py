import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
work_dir = 'logs/tiny_airsim/'
log_csv = load_file(work_dir+'log.csv')
log_csv = [[float(element) for element in row] for row in log_csv[1:]]
mat_csv = np.array(log_csv[1:])
epochs = mat_csv[:, 0]
loss = mat_csv[:, 1]
lr = mat_csv[:, 2]
val_loss = mat_csv[:, 3]

n = len(epochs)
mAP = np.zeros((n,))
for i in range(n):
    file = load_file(work_dir+'results/industrial_room--{}.pkl'.format(i))
    results = [ResultByConfidence(d) for d in file['results']['MetricDetection']]
    avg_precision, recall = average_precision_recall(results)
    mAP[i] = np.mean(avg_precision)

training_plot = BaseMultiPlot(x_data=[epochs, epochs, epochs, epochs], x_label='Epochs',
                              y_data=[mAP, loss/10, lr*100, val_loss/10], legend=['mAP', 'loss/10', 'lr*100', 'val_loss/10'],
                              y_lim=(0.0, 1.0),
                              title='Training')
training_plot.show()
