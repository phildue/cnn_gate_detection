import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.thesis.plot import plot_result
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'objectdetect/yolov3_w0_416x416',
    'objectdetect/yolov3_w1_416x416',
    'objectdetect/yolov3_w2_416x416',
    'objectdetect/yolov3_w3_416x416',
    'objectdetect/yolov3_d0_416x416',
    'objectdetect/yolov3_d1_416x416',
    'objectdetect/yolov3_d2_416x416',
    'datagen/yolov3_blur416x416',
    'datagen/yolov3_arch2416x416',

]

work_dir = 'out/thesis/'
n_iterations = 2

names = [
    'W0',
    'W1',
    'W2',
    'W3',
    'D0',
    'D1',
    'D2',
    'TinyYoloV3',
    'Arch2'
]
frame = plot_result(models=models, names=names, work_dir=work_dir, n_iterations=n_iterations)
print(frame.to_string())
print(frame.to_latex())
plt.show(True)
