import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()

models = [
    'mavnet',
    'mavnet_lowres320',
    'mavnet_lowres160',
    'mavnet_lowres80',
]

img_res = [
    416 * 416,
    320 * 320,
    160 * 160,
    80 * 80,
]

legend = [
    '416x416',
    '320x320',
    '160x160',
    '80x80',
]

iou = 0.6
n_iterations=2
frames = []
for m in models:
    frame = None
    for it in range(n_iterations):
        try:
            new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_iros2018_course_final_simple_17gates/results_size_cluster.pkl'.format(m, it))
            if frame is None:
                frame = new_frame
            else:
                frame.merge(new_frame)
        except FileNotFoundError as e:
            print(e)
            continue
    print(frame.to_string())
    frames.append(frame)

bins = frames[0]['Sizes Bins']
print(frame.to_string())
plt.figure(figsize=(8, 3))
plt.title('Performance for Bins of Different Object Areas')
w = 1.0 / len(frame['Name'])
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
for i_m, r in enumerate(models):
    aps = []
    for it in range(n_iterations):
        try:
            ap = frames[i_m]['AveragePrecision{0:02f}_i{1:02d}'.format(iou, it, 'test_basement_cats')]
            aps.append(ap)
        except KeyError as e:
            print(e)
    ap = np.mean(aps, 0)
    err = np.std(aps, 0)

    plt.bar(np.arange(len(bins)) - len(models) * w * 0.5 + i_m * w, ap, width=w, yerr=err)


plt.xlabel('Area Relative to Image Size')
plt.ylabel('Average Precision')
plt.xticks(np.arange(len(bins)), np.round(bins, 3))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                     wspace=0.4, hspace=0.4)
plt.legend(legend)
plt.ylim(0, 1.1)
plt.savefig('doc/thesis/fig/perf_resolution_size.png')

# plt.figure(figsize=(8, 3))
# plt.title('AveragePrecision')
# w = 1.0 / (2 * len(titles))
# n_true = np.array(frame['Objects'][0])  # / np.sum(frame['Objects'][0])
# legend = []
# w = 0.5
# for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
#     legend.append(titles[i])
#     legend.append(titles[i] + ' balanced')
#     plt.bar(frame['Layers'][i]-0.5*w, frame['AP Total' + str(iou)][i], width=w)
#     plt.bar(frame['Layers'][i] + 0.5 * w, np.sum(r * 1 / n_true), width=w)
#
#     plt.xlabel('Layers')
#     plt.ylabel('Average Precision')
#     plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
#                         wspace=0.4, hspace=0.4)
# plt.legend(legend)

# plt.savefig('doc/thesis/fig/depth_ap_size.png')

plt.show(True)
