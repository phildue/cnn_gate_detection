from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
analyzer = SetAnalyzer((416, 416), ['resource/ext/samples/random_flight/'])
labels = analyzer.labels
objs = []
import numpy as np

img = Image(np.zeros((416, 416, 3)), 'bgr')

for l in labels:
    for o in l.objects:
        if o.width > 416 or o.height > 416:
            show(img, labels=l)
poses = [o.pose for o in objs]

norths = [p.north for p in poses]

weird = []
