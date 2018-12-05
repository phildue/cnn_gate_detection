import matplotlib.pyplot as plt
import numpy as np

from utils.SetAnalysis import SetAnalysis
from utils.workdir import cd_work

cd_work()

img_shape = (416, 416)
path = 'resource/ext/samples/'
sets = [
    'random_iros',
    'iros2018_course_final_simple_17gates',

]
titles = [
    'Random Placement',
    'Simulated Flight',
]
areas = []
label_maps = []
aspect_ratios = []
pose_cluster = []
poses = []
angles = np.linspace(-360, 360, 6)
distances = np.linspace(0, 10, 3)
for s in sets:
    set_analyzer = SetAnalysis(img_shape, path + s)
    area = set_analyzer.get_area()
    # area_filtered = area[(0 < area) & (area < 2.0)]
    # print("Removed: {}".format(len(area) - len(area_filtered)))
    areas.append(area)
    label_maps.append(set_analyzer.get_label_map())
    box_dims = set_analyzer.get_box_dims()
    poses.append(set_analyzer.get_poses())
    aspect_ratios.append(box_dims)
    pose_cluster.append(set_analyzer.pose_cluster(angles, distances))

plt.figure(figsize=(8, 3))
plt.title("Heatmap of Object Appearances", fontsize=12)
for i, m in enumerate(label_maps, 1):
    plt.subplot(1, 2, i)
    plt.pcolor(m,cmap=plt.cm.viridis, vmin=0, vmax=250)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.colorbar()

    plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.show(False)
plt.savefig('doc/thesis/fig/heatmap_camplace.png')

plt.figure(figsize=(8, 3))
plt.title("Histogram of Object Locations", fontsize=12)
for i in range(len(sets)):
    plt.subplot(1, 2, i+1)
    plt.title(titles[i], fontsize=12)
    yaws = np.array([np.degrees(p.yaw) for p in poses[i]])
    yaws[yaws < 0] *= -1
    yaws[yaws > 180] = 360 - yaws[yaws > 180]
    yaws = 180 - yaws
    yaws[yaws > 90] = 180 - yaws[yaws > 90]
    dists = [np.linalg.norm(p.transvec / 3) for p in poses[i]]
    plt.hist2d(yaws, dists, bins=10,cmap=plt.cm.viridis, vmin=0, vmax=100)
    # plt.ylim(0,12)
    plt.colorbar()
    plt.xlabel("$\Delta \psi$")
    plt.ylabel("$|\Delta t|$")

    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                                            wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/hist2d_camplace.png')
plt.show(True)
