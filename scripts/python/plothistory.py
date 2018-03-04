from backend.visuals.plots.PlotTrainingHistory import PlotTrainingHistory
from fileaccess.utils import load_file
from workdir import work_dir

work_dir()

src_dir = 'logs/yolov2_10k_new/'

history = load_file(src_dir + 'training_history.pkl')
PlotTrainingHistory(history).show()
