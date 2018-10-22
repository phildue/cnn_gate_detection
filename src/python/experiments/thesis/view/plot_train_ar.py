from utils.fileaccess.labelparser.DatasetParser import DatasetParser
import matplotlib.pyplot as plt
import pandas as pd

from utils.workdir import cd_work

cd_work()
datasets = [['resource/ext/samples/various_environments20k'],
            ['resource/ext/samples/daylight_course1',
             'resource/ext/samples/daylight_course5',
             'resource/ext/samples/daylight_course3',
             'resource/ext/samples/iros2018_course1',
             'resource/ext/samples/iros2018_course5',
             'resource/ext/samples/iros2018_flights',
             'resource/ext/samples/basement_course3',
             'resource/ext/samples/basement_course1',
             'resource/ext/samples/iros2018_course3_test']]
names = ['Random',
         'Flight']
frame = pd.DataFrame()
frame['Name'] = names
widths = []
heights = []
for i, d in enumerate(datasets, 0):

    labels = []
    for p in d:
        label_reader = DatasetParser.get_parser(p, 'xml', color_format='bgr')
        img_labels = label_reader.read(1000)[1]
        for l in img_labels:
            labels.extend(l.objects)

    ws = [l.poly.width for l in labels if l.poly.width < 416*1.5 and l.poly.height < 416*1.5]
    hs = [l.poly.height for l in labels if l.poly.width < 416*1.5 and l.poly.height < 416*1.5]
    widths.append(ws)
    heights.append(hs)

frame['Width'] = widths
frame['Height'] = heights
print(frame.to_string())
plt.figure(figsize=(8, 3))

for i, d in enumerate(datasets):
    plt.subplot(1, 2, i+1)
    plt.title(names[i])
    plt.hist2d(frame['Width'][i], frame['Height'][i], bins=10, cmap=plt.cm.viridis)
    # plt.ylim(0,12)
    plt.colorbar()
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.3, hspace=0.3)

plt.show()
