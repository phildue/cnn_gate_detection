import matplotlib.pyplot as plt

from experiments.thesis.plot import plot_result
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_allview416x416',
    # 'yolov3_blur416x416',
    # 'yolov3_chromatic416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    'Random View Points',
    'Flight',
    'Combined'
    # 'Flight + Random Blur',
    # 'Flight + Random Chrom',
]
frame = plot_result(models=models, names=names, work_dir=work_dir, n_iterations=n_iterations)
print(frame.to_string())
print(frame.to_latex())
plt.savefig('doc/thesis/fig/recall_bar.png')

plt.show(True)
