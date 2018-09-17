import matplotlib.pyplot as plt

from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from utils.workdir import cd_work

cd_work()

img_shape = (480, 640)
path = 'resource/ext/samples/'
sets = [
    'real_test_labeled',
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway'
]
titles = [
    'Total',
    'Cyberzoo',
    'Basement',
    'Hallway'
]
areas = []
label_maps = []
for s in sets:
    set_analyzer = SetAnalyzer(img_shape, path + s)
    areas.append(set_analyzer.get_area())
    label_maps.append(set_analyzer.get_label_map())

plt.figure(figsize=(8, 6))
plt.suptitle("Heatmap of Bounding Box Locations", fontsize=12)
for i, m in enumerate(label_maps, 1):
    plt.subplot(2, 2, i)
    plt.imshow(m)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.colorbar()
    plt.title(titles[i - 1], fontsize=12)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.show(False)
plt.savefig('doc/thesis/fig/heatmap_real.png')
plt.figure(figsize=(8, 6))
plt.suptitle("Histogram of Object Sizes", fontsize=12)
for i, a in enumerate(areas, 1):
    plt.subplot(2, 2, i)
    plt.hist(a, bins=10, rwidth=0.5)
    plt.xlabel("Area as Fraction of Total Size", fontsize=12)
    plt.ylabel("Number of Objects", fontsize=12)
    plt.title(titles[i - 1], fontsize=12)
    plt.xlim(0, 1.0)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.5)
plt.savefig('doc/thesis/fig/histogram_real.png')
plt.show(False)
