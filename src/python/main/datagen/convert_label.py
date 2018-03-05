from fileaccess.LabelFileParser import LabelFileParser
from labels.GateLabel import GateLabel
from labels.ObjectLabel import ObjectLabel
from workdir import work_dir

from src.python.utils.labels.ImgLabel import ImgLabel

work_dir()

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
