import os
import sys
from os.path import expanduser


PROJECT_ROOT = expanduser('~') + '/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from SetAnalyzer import SetAnalyzer


path = 'resource/samples/mult_gate_valid/'
img_shape = (416, 416)
heat_map = SetAnalyzer(img_shape, path).get_heat_map()
heat_map.show()
