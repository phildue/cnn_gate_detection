from keras import Model

from modelzoo.backend.tensor.gatenet.PostprocessLayer import PostprocessLayer
from modelzoo.models.ModelFactory import ModelFactory
from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
model = ModelFactory.build('gatev10', src_dir='out/gatev10_mixed/')

keras_model = model.net.backend
out = keras_model.layers[-1].output
input = keras_model.layers[0].input
postprocessed = PostprocessLayer()(out)

inference_model = Model(input, postprocessed)

generator = GateGenerator(directories=['resource/ext/samples/daylight_flight'],
                          batch_size=100, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0,
                          label_format='xml',
                          )
n_samples = 100
for i in range(int(n_samples / generator.batch_size)):
    batch = next(generator.generate())
    for j in range(len(batch)):
        img = batch[j][0]
        label = batch[j][1]
        img, label = resize(img, model.input_shape, label=label)
        img_enc = model.preprocessor.preprocess(img)
        out = inference_model.predict(img_enc)[0]
        boxes = BoundingBox.from_tensor_minmax(out[:, 4:], out[:, :4])
        boxes = [b for b in boxes if b.class_conf > 0.6]
        label_pred = BoundingBox.to_label(boxes)
        show(img, 'demo', labels=[label_pred, label], t=1)
