import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
frame = pd.read_pickle('out/results/size_sim.pkl')
print(frame.to_string())
plt.figure(figsize=(8, 3))
plt.title('Performance for Bins of Different Object Areas')
w = 1.0 / len(frame['Name'])
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
legend = []
for i, r in enumerate(frame['AveragePrecision' + str(0.6)]):
    plt.bar(np.arange(len(r)) - len(frame['Name']) * w * 0.5 + i * w, r, width=w)
    legend.append(str(frame['Layers'][i]) + ' Layers')
    plt.xlabel('Area Relative to Image Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(len(r)), np.round(frame['Sizes Bins'][i], 3))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(frame['Name'])
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
