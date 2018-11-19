import ezodf
import matplotlib.pyplot as plt
import pandas as pd

from utils.workdir import cd_work


def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header].value: [x.value for x in col[header + 1:]]
                         for col in tab.columns()})


cd_work()

frame = read_ods('doc/results/speed.ods', 1)

print(frame.to_string())
res = ['20x15', '160x120', '80x60', '320x240']
style = ['x--', 'o--', '<--', '>--']

plt.figure(figsize=(8, 4))
handles = []
for i, r in enumerate(res):
    x = frame['Total Operations'][frame['Resolution'] == r]
    y = frame['Inference Time'][frame['Resolution'] == r]
    err = frame['Variance'][frame['Resolution'] == r]
    l = sorted(zip(x, y, err))
    y = [y for _, y, _ in l]
    x = [x for x, _, _ in l]
    err = [err for _, _, err in l]
    # h = plt.plot(x, y,style[i])
    h = plt.errorbar(x, y, err, fmt=style[i], elinewidth=1, capsize=2)

    handles.append(h[0])

plt.legend(handles, res, numpoints=1)
plt.title('Speed Measurements on JeVois Smart Camera')
plt.xlabel('Total Operations in Layer')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ylabel('Inference Time [ms]')
plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-')
plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')
plt.minorticks_on()
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/bottleneck_jevois.png')
plt.show(True)
