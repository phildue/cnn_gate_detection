from src.python.utils.workdir import work_dir

work_dir()

from src.python.utils.fileaccess import load_file
from src.python.modelzoo.evaluation import StreamAnalyzer

result_path = 'logs/tinyyolo_10k/stream3/'
result_file = 'result.pkl'
results = load_file(result_path + result_file)

analyzer = StreamAnalyzer(result_path+result_file)
analyzer.loc_error_plot().show()

