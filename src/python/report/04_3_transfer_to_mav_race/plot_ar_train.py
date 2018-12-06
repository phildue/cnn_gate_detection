import matplotlib.pyplot as plt

from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.labels.ObjectLabel import ObjectLabel
from utils.timing import tic, tuc, toc
from utils.workdir import cd_work

cd_work()

sets_race = ['resource/ext/samples/train_basement_gate',
             'resource/ext/samples/daylight_course5',
             'resource/ext/samples/daylight_course3',
             'resource/ext/samples/iros2018_course1',
             'resource/ext/samples/iros2018_course5',
             'resource/ext/samples/iros2018_flights',
             'resource/ext/samples/basement_course3',
             'resource/ext/samples/basement_course1',
             'resource/ext/samples/iros2018_course3_test',
             # 'resource/ext/samples/various_environments20k',
             # 'resource/ext/samples/realbg20k'
             ]

sets_random = [
    'resource/ext/samples/various_environments20k',
]

sets_combined = sets_race + sets_random
titles = [
    'Random View',
    'Drone Model',
    'Combined'
]

ObjectLabel.classes = ['gate']


def scatter_plot(sets, step, style):
    for d in sets:
        labels = DatasetParser.get_parser(d, 'xml', 'bgr').read(2000)[1]
        for l in labels[::step]:
            for o in l.objects:
                x = o.poly.width
                y = o.poly.height
                if 0 < x < 416 and 0 < y < 416:
                    h = plt.plot(x, y, style)
        tuc()
    return h


tic()
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.title('Random Placement')
h1 = scatter_plot(sets_random, 4, 'b.')
plt.xlabel("$b_w$")
plt.ylabel("$b_h$")
toc()

# tic()
plt.subplot(1, 2, 2)
plt.title('Simulated Flight')
h2 = scatter_plot(sets_race, 4, 'b.')
plt.xlabel("$b_w$")
plt.ylabel("$b_h$")
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
# toc()


plt.savefig('doc/thesis/fig/ar_train.png')
plt.show(True)
