import os
import sys

from backend.visuals import plot_training_history

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

PROJECT_ROOT = '/home/phil/dronevision'
os.chdir(PROJECT_ROOT)

from models.Yolo.TinyYolo import TinyYolo
from fileaccess.VocGenerator import VocSetParser

BATCH_SIZE = 8
result_path = 'logs/tiny-yolo-voc/'
model = TinyYolo(batch_size=BATCH_SIZE)
trainset = VocSetParser("resource/samples/VOCdevkit/VOC2012/Annotations/",
                        "resource/samples/VOCdevkit/VOC2012/JPEGImages/", batch_size=BATCH_SIZE, n_samples=32)

train_params = {'optimizer': 'adam',
                'lr': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.0005,
                'initial_epoch': 0,
                'epochs': 1,
                'log_dir': result_path}

model.fit_generator(trainset, params=train_params, out_file=result_path + 'test.h5')

plot_training_history(model, show=False, output_file=result_path + 'test_history.png')
#plot_model(model, output_file=result_path + 'model.png')
#write_report(title='tiny-yolo-voc',output_file=result_path+'report.tex',train_params)