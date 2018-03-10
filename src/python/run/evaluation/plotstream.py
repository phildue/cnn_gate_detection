from modelzoo.evaluation.StreamAnalyzer import StreamAnalyzer
from utils.fileaccess.utils import load_file
from utils.workdir import work_dir

work_dir()


result_path = 'logs/tinyyolo_10k/stream3/'
result_file = 'result.pkl'
results = load_file(result_path + result_file)

analyzer = StreamAnalyzer(result_path+result_file)
analyzer.loc_error_plot().show()

