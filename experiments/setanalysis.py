from timing import tic, toc

from SetAnalyzer import SetAnalyzer
from workdir import work_dir

work_dir()
tic()
path = 'resource/samples/mult_gate_valid/'
img_shape = (416, 416)
heat_map = SetAnalyzer(img_shape, path).get_heat_map()
toc()
heat_map.show()
