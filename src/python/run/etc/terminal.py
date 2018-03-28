from utils.workdir import work_dir
import numpy as np
import keras.backend as K
from utils.fileaccess.utils import *

work_dir()

print(load_file('logs/tiny_bebop/results/cyberzoo--0.pkl'))
