import argparse

from evaluation.evaluation import infer_detector_on_set
from modelzoo.Decoder import Decoder
from modelzoo.Detector import Detector
from modelzoo.Encoder import Encoder
from modelzoo.Postprocessor import Postprocessor
from modelzoo.Preprocessor import Preprocessor
from modelzoo.build_model import build_detector
from utils.ModelSummary import ModelSummary
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

show_t = 1
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--show', type=int, default=show_t)
args = parser.parse_args()
show_t = args.show
cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    # 'distortion',
    # 'blur',
    # 'hsv',
    # 'exposure',
    # 'chromatic',
    # 'mavlabgates'
    'blur_distortion',
]

datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]

preprocessing= [
TransformResize((240, 320))
]

work_dir = 'out/'
n_iterations = 2
ObjectLabel.classes = ['gate']
for dataset in datasets:
    for i_m, model in enumerate(models):
        for i in range(0, n_iterations):
            model_folder = model + '_i0{}'.format(i)
            try:

                summary = ModelSummary.from_file(work_dir + model_folder + '/summary.pkl')
                anchors = summary.anchors/2
                img_res = (240, 320)
                architecture = summary.architecture
                color_format = summary.color_format
                network, output_grids = build_detector(img_shape=(img_res[0], img_res[1], 3), architecture=[
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1},
                    # {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 8, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'predict'},
                    {'name': 'route', 'index': [10]},
                    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'upsample', 'size': 2},
                    {'name': 'crop', 'top': 1, 'bottom': 0, 'left': 0, 'right': 0},
                    {'name': 'route', 'index': [-1, 9]},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'predict'},
                    {'name': 'route', 'index': [10]},
                    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'upsample', 'size': 4},
                    {'name': 'crop', 'top': 2, 'bottom': 0, 'left': 0, 'right': 0},
                    {'name': 'route', 'index': [-1, 7]},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'predict'}
                ],
                                                       anchors=anchors,
                                                       n_polygon=4)
                network.load_weights(work_dir + model_folder + '/model.h5')
                encoder = Encoder(anchor_dims=anchors, img_norm=img_res, grids=output_grids, n_polygon=4, iou_min=0.4)
                preprocessor = Preprocessor(preprocessing=preprocessing, encoder=encoder, n_classes=1,
                                            img_shape=img_res,
                                            color_format=color_format)
                decoder = Decoder(anchor_dims=anchors, n_polygon=4, norm=img_res, grid=output_grids)
                postproessor = Postprocessor(decoder=decoder)

                detector = Detector(network, preprocessor, postproessor, summary)

                infer_detector_on_set(
                    detector=detector,
                    result_path=work_dir + model_folder + '/test_' + dataset + '/',
                    result_file='predictions320',
                    batch_size=4,
                    show_t=show_t,
                    preprocessing=preprocessing,
                    image_source=['resource/ext/samples/{}/'.format(dataset)])
            except FileNotFoundError:
                print("Not found: {}".format(model_folder))
