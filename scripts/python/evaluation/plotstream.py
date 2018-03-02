from workdir import work_dir

work_dir()

from fileaccess.utils import load_file
from frontend.evaluation.StreamAnalyzer import StreamAnalyzer

result_path = 'logs/tinyyolo_10k/stream3/'
result_file = 'result.pkl'
results = load_file(result_path + result_file)

analyzer = StreamAnalyzer(result_path+result_file)
analyzer.loc_error_plot().show()

