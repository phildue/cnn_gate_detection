from fileaccess.LabelFileParser import LabelFileParser

from workdir import work_dir

work_dir()

parser = LabelFileParser('resource/samples/mult_gate_aligned', '.pkl')
labels = parser.read()
parser.write(labels)
