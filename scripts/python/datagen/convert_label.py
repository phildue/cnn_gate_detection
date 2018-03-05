from labels.ObjectLabel import ObjectLabel

from fileaccess.LabelFileParser import LabelFileParser

from workdir import work_dir

work_dir()

parser = LabelFileParser('resource/samples/mult_gate_aligned_blur_distort', 'pkl')
labels = parser.read()

ObjectLabel.classes = 'gate'
for l in labels:
    for o in l.objects:
        o.class_name = 'gate'
parser.write(labels)
