from modelzoo.evaluation import evaluate_generator, evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.workdir import cd_work
import numpy as np

cd_work()

name = 'industrial'

# Image Source
batch_size = 8
n_batches = int(100 / batch_size)
image_source = ['resource/ext/samples/industrial_new_test/']
color_format = 'bgr'

# Model
model_src = 'out/refnet52x52->3x3+4layers+32filters/'
conf_thresh = 0
summary = load_file(model_src + 'summary.pkl')
architecture = summary['architecture']
img_res = 52, 52
grid = 3, 3
model = GateNet.create_by_arch(norm=img_res, architecture=architecture,
                               anchors=np.array([[[1, 1],
                                                  [1 / grid[0], 1 / grid[1]],  # img_h/img_w
                                                  [2 / grid[0], 2 / grid[1]],  # 0.5 img_h/ 0.5 img_w
                                                  [1 / grid[0], 3 / grid[0]],  # img_h / 0.33 img_w
                                                  [1 / grid[0], 2 / grid[0]]  # img_h / 0.5 img_w
                                                  ]]),
                               batch_size=batch_size,
                               color_format='yuv',
                               conf_thresh=conf_thresh,
                               augmenter=None,
                               weight_file=model_src + 'model.h5'
                               )

# Evaluator
iou_thresh = 0.8

# Result Paths
result_path = model_src
result_file = 'result_' + name + str(iou_thresh)
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + str(iou_thresh) + '.txt'

create_dirs([result_path, result_img_path])
generator = CropGenerator(GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                                        shuffle=False, color_format=color_format, label_format='xml', start_idx=0))

evaluate_generator(model, generator, n_batches=n_batches, verbose=True, out_file_labels=result_path + result_file)

evaluate_file(model, result_path + result_file + '.pkl', metrics=[MetricDetection(iou_thresh=iou_thresh, show_=False)],
              verbose=True,
              out_file_metric=result_path + result_file + '.pkl')

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'train_samples': '10k',
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * batch_size}

save_file(exp_params, exp_param_file, result_path)
