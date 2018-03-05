from labels.ImgLabel import ImgLabel

from labels.GateLabel import GateLabel

from labels.ObjectLabel import ObjectLabel

from fileaccess.LabelFileParser import LabelFileParser

from workdir import work_dir

work_dir()

parser = LabelFileParser('resource/samples/mult_gate_aligned_blur_distort', 'pkl')
labels = parser.read()

ObjectLabel.classes = 'gate'
labels_conv = []
for l in labels:
    obj_conv = []
    for o in l.objects:
        obj_conv.append(GateLabel(o.position, o.gate_corners, o.confidence, 'gate'))
    labels_conv.append(ImgLabel(obj_conv))
parser.write(labels_conv)
