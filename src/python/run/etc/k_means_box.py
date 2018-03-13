from sklearn.cluster import KMeans

from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
import numpy as np

from utils.workdir import work_dir

work_dir()
generator = GateGenerator(directories=['resource/ext/samples/bebop/'],
                          batch_size=10, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0,
                          label_format='xml')
boxes_centered = []
idx = 0
for batch in generator.generate():
    for img, label, _ in batch:
        boxes = BoundingBox.from_label(label)

        for b in boxes:
            b.cx = 0
            b.cy = 0

        boxes_centered.extend(boxes)
    if idx > generator.n_samples:
        break
    idx += 1

ious = []
for b_0 in boxes_centered:
    for b_1 in boxes_centered:
        iou = b_0.iou(b_1)
        ious.append(iou)
ious = np.array(ious)
ious = np.reshape(ious, [-1, 1])

kmeans = KMeans().fit(ious)
print("Centers", kmeans.cluster_centers_)
