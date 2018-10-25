import matplotlib.pyplot as plt

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
    'Racing Track',
]
areas = []
label_maps = []
aspect_ratios = []
for s in sets:
    set_analyzer = SetAnalysis(img_shape, path + s)
    area = set_analyzer.get_area()
    area_filtered = area  [(0 < area) & (area < 2.0)]
    print("N = {}".format(len(area_filtered)))
    print("Removed: {}".format(len(area) - len(area_filtered)))
    areas.append(area_filtered)
    label_maps.append(set_analyzer.get_label_map())
    box_dims = set_analyzer.get_box_dims() [(0 < area) & (area < 2.0)]

    aspect_ratios.append(box_dims)

plt.figure(figsize=(8, 3))
plt.title("Heatmap of Bounding Boxes", fontsize=12)
for i, m in enumerate(label_maps, 1):
    plt.subplot(1, 2, i)
    plt.imshow(m)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.colorbar()

    plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.show(False)
plt.savefig('doc/thesis/fig/heatmap_camplace.png')
plt.figure(figsize=(8, 3))
plt.title("Histogram of Bounding Box Sizes", fontsize=12)
for i, a in enumerate(areas, 1):
    plt.subplot(1, 2, i)
    plt.hist(a, bins=10, rwidth=0.5)
    plt.xlabel("Area as Fraction of Total Size", fontsize=12)
    plt.ylabel("Number of Boxes", fontsize=12)
    plt.xlim(0, 2.0)
    plt.ylim(0, 50)
    plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                   wspace=0.3, hspace=0.5)
plt.savefig('doc/thesis/fig/histogram_camplace.png')
plt.show(False)

plt.figure(figsize=(8, 3))
plt.title("Aspect Ratios", fontsize=12)
for i, a in enumerate(aspect_ratios, 1):
    plt.subplot(1, 2, i)
    plt.plot(a[:, 0], a[:, 1], 'rx')
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.xlim(0, 2.0)
    plt.ylim(0, 2.0)
    plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                   wspace=0.3, hspace=0.5)
plt.savefig('doc/thesis/fig/aspect_ratios_camplace.png')
plt.show(False)

plt.figure(figsize=(8, 3))
plt.title("Aspect Ratio over Size", fontsize=12)
for i, a in enumerate(aspect_ratios, 1):
    plt.subplot(1, 2, i)
    plt.plot(a[:, 1] / a[:, 0], areas[i - 1], 'rx')
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    # plt.xlim(0, 10)
    # plt.ylim(0, 1)
    # plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                   wspace=0.3, hspace=0.5)
# plt.savefig('doc/thesis/fig/histogram_random_view.png')
plt.show(True)
