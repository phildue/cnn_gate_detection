from workdir import work_dir

from src.python.modelzoo.backend.visuals import PlotTrainingHistory
from src.python.utils.fileaccess import load_file

work_dir()

src_dir = 'logs/yolov2_10k_new/'

history = load_file(src_dir + 'training_history.pkl')
PlotTrainingHistory(history).show()
