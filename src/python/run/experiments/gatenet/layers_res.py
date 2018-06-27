# How does performance compare over resolution, width, depth
from run.training.gate.train import train
import numpy as np

from utils.workdir import cd_work

cd_work()
img_ress = [(104, 104), (52, 52)]
for i, img_res in enumerate(img_ress):
    for width in [64, 32, 16]:

        baseline = int(np.log2(img_res[0]/13)) * [
            {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)}]

        for n_layers in range(8-int(len(baseline)/2)):
            architecture = baseline.copy()

            for j in range(n_layers):
                architecture.append(
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1})

            train(architecture=architecture,
                  work_dir='gatenet{}x{}+{}layers+{}filters'.format(img_res[0], img_res[1],
                                                                    len(architecture) - int(len(baseline) / 2), width),
                  img_res=img_res,
                  epochs=50)
