import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'sign',
    'cats',
    'ewfo',
]

datasets = [
    'test_basement_gate',
    'test_basement_sign',
    'test_basement_cats',
    'test_iros_gate',
    'test_iros_cats',
    'test_iros_sign',
]

legend = [
    'Trained on Sign',
    'Trained on Cats',
    'Trained on EWFO',
]

n_iterations = 1
frames = []
for m in models:
    frame = None
    for d in datasets:
        for it in range(n_iterations):
            try:
                new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_size_cluster.pkl'.format(m, it, d))
                # if frame is None:
                frame = new_frame
                # else:
                #     frame.merge(new_frame)

            except FileNotFoundError as e:
                print(e)
                continue
    print(frame.to_string())
    frames.append(frame)

bins = frames[0]['Sizes Bins']

plt.figure(figsize=(16, 9))
i_p = 0
for d in datasets:
    i_p += 1
    plt.subplot(3, 2, i_p)
    plt.title('Tested on {}'.format(d))
    w = 1.0 / len(models)
    # plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')

    for i_m, r in enumerate(models):

        aps = []
        for it in range(n_iterations):
            try:
                ap = frames[i_m]['{2:s}_ap{0:02f}_i{1:02d}'.format(0.6, it, d)]
                aps.append(ap)
            except KeyError as e:
                print(e)
        ap = np.mean(aps, 0)
        err = np.std(aps, 0)

        plt.bar(np.arange(len(bins)) + i_m * w, ap, width=w, yerr=err)

    for x, y in enumerate(frames[0]['{} Objects'.format(d)]):
        plt.text(x - 0.1, 1.0, str(np.round(y, 2)), color='gray', fontweight='bold')
    plt.xlabel('Area Relative to Image Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(len(bins)), np.round(bins, 3))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
    plt.legend(legend, loc='lower left')
    plt.ylim(0, 1.1)
    plt.savefig('doc/thesis/fig/basement_cats_size.png')

plt.show(True)
