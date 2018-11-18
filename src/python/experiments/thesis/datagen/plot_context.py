import matplotlib.pyplot as plt

from experiments.thesis.plot import plot_result
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_gate_realbg416x416',
    'yolov3_gate_uniform416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_mixed416x416',
    'yolov3_gate416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 2

names = [
    'Real Backgrounds',
    'Uniform Backgrounds',
    'Various Environments',
    'Real + Various',
    'Single Background'
]
frame = plot_result(models=models, names=names, work_dir=work_dir, n_iterations=n_iterations,validation_set=True)
print(frame.to_string())
print(frame.to_latex())
# plt.savefig('doc/thesis/fig/context_bar.png')
plt.show(True)
