import numpy as np

from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
result_file = 'logs/yolov2_10k/speed_server/result.pkl'

content = load_file(result_file)

results = content['results']

fps_total = np.array([e['fps_total'] for e in results])
fps_enc = np.array([e['fps_enc'] for e in results])
fps_pp = np.array([e['fps_pp'] for e in results])
fps_pred = np.array([e['fps_pred'] for e in results])

print(np.mean(1 / fps_total))
print(np.std(1 / fps_total))

print(np.mean(1 / fps_pred) / np.mean(1 / fps_total))

print(np.mean(fps_total))
print(np.std(fps_total))
