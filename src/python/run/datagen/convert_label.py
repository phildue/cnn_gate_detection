import sys

from utils.fileaccess.LabelFileParser import LabelFileParser
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()
sys.path.append('utils/labels')
parser = LabelFileParser('resource/samples/cyberzoo_conv/', 'pkl')
labels = parser.read()

ObjectLabel.classes = 'gate'
labels_conv = []
for l in labels:
    obj_conv = []
    for o in l.objects:
        obj_conv.append(GateLabel(o.position, o.gate_corners, o.confidence, 'gate'))
    labels_conv.append(ImgLabel(obj_conv))
parser.write(labels_conv)
