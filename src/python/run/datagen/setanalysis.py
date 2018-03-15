from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from utils.timing import tic, toc
from utils.workdir import work_dir

work_dir()
tic()
path = 'resource/ext/samples/bebop20k/'
img_shape = (180, 315)
set_analyzer = SetAnalyzer(img_shape, path)
toc()
set_analyzer.show_summary()
