from workdir import work_dir

work_dir()

from fileaccess.utils import load
from evaluation.StreamAnalyzer import StreamAnalyzer

result_path = 'logs/tinyyolo-noaug/stream1/'
result_file = 'result.pkl'
results = load(result_path + result_file)

analyzer = StreamAnalyzer(result_path+result_file)
analyzer.loc_error_plot().show()
analyzer.detection_eval()

