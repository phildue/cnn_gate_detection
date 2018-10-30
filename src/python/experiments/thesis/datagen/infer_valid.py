from modelzoo.evaluation.evalset import evalset
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

# models = [name for name in os.listdir('out/0108/')]
models = [
    # 'yolov3_gate_realbg416x416',
    # 'yolov3_gate_uniform416x416',
    # 'yolov3_gate_dronemodel416x416',
    'yolov3_gate416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_mixed416x416',
    # 'yolov3_allgen416x416',
    # 'yolov3_hsv416x416',
    # 'yolov3_blur416x416',
    # 'yolov3_chromatic416x416',
    # 'yolov3_exposure416x416',
    # 'yolov3_40k416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 2
ObjectLabel.classes = ['gate']
exp_name = 'datagen'
for model in models:
    for i in range(1, n_iterations):
        model_folder = model + '_i0{}'.format(i)
        prediction_file = 'predictions'
        training_set = load_file(work_dir + model_folder + '/summary.pkl')['image_source']
        evalset(name=exp_name,
                result_path=work_dir + model_folder + '/test_valid/',
                result_file=prediction_file,
                batch_size=4,
                n_samples=100,
                model_src=work_dir + model_folder,
                preprocessing=None,
                color_format='bgr',
                image_source=training_set)
