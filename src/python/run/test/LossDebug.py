import os
import sys

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from src.python.modelzoo.models.yolo import Yolo
from src.python.modelzoo.models import YoloPreprocessor
from src.python.utils.fileaccess import VocGenerator
import keras.backend as K

anchors = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
anchors = [float(anchors.strip()) for anchors in anchors.split(',')]
scale_conf = 5.0
scale_coor = 5.0
scale_prob = 1.0
scale_noob = 0.5
n_classes = 20
preprocessor = YoloPreprocessor()

dataset = VocGenerator("resource/samples/VOCdevkit/VOC2012/Annotations/",
                       "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=100).generate(100)

model_yadk = Yolo(grid=(13, 13), norm=(416, 416),
                  scale_noob=scale_noob,
                  scale_conf=scale_conf,
                  scale_coor=scale_coor,
                  scale_prob=scale_prob,
                  anchor_file=None,
                  yadk_model=True, conf_thresh=0.3)

loss = model_yadk.get_loss()

batch = dataset.__next__()
idx = 20  # random.randint(0, 99)

x_true, y_true = preprocessor.preprocess_img(batch[idx], augment=False)

sample_vec = K.np.expand_dims(x_true, axis=0)
netout = model_yadk.net.predict(sample_vec)
netout = K.np.reshape(netout, [netout.shape[0], 13, 13, 5, n_classes + 5])

y_ntf = model_yadk.net2y(netout[0])
sess = K.tf.InteractiveSession()
y_tf = loss.net2y(netout)

y_tf = y_tf.eval()
print(K.np.linalg.norm(y_ntf - y_tf))
