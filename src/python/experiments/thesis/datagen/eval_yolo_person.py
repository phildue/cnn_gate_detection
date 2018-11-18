from evaluation import evalmetric
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = ['yolov3_person416x416']

work_dir = 'out/thesis/datagen/'
n_iterations = 1
ObjectLabel.classes = ['muro']
for model in models:
    for i in range(n_iterations):
        model_folder = model + '_i0{}'.format(i)
        # evalset(name='complex_vs_efo',
        #         result_path=work_dir + model_folder + '/test/',
        #         result_file='predictions_iros2018_random_test.pkl',
        #         batch_size=16,
        #         model_src=work_dir + model_folder,
        #         preprocessing=None,
        #         color_format='bgr',
        #         image_source=['resource/ext/samples/iros2018muro_random_test/'])

        for iou_thresh in [0.4, 0.6, 0.8]:
            evalmetric(name='total_iou{}'.format(iou_thresh),
                       min_box_area=0,
                       max_box_area=1.2,
                       iou_thresh=iou_thresh,
                       model_src=work_dir + model_folder,
                       label_file=work_dir + model_folder + '/test/predictions_iros2018_random_test.pkl',
                       result_path=work_dir + model_folder + '/test/',
                       show=False)
