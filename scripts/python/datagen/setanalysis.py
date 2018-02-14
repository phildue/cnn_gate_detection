from timing import tic, toc

from SetAnalyzer import SetAnalyzer
from workdir import work_dir

work_dir()
tic()
path = 'samplegen/resource/shots/mult_gate_aligned/'
img_shape = (416, 416)
heat_map = SetAnalyzer(img_shape, path).get_heat_map()
toc()
heat_map.show()
