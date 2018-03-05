from samplegen.SetAnalyzer import SetAnalyzer
from utils.timing import tic, toc
from utils.workdir import work_dir

work_dir()
tic()
path = 'resource/samples/mult_gate_aligned/'
img_shape = (416, 416)
heat_map = SetAnalyzer(img_shape, path).get_heat_map()
toc()
heat_map.show()
