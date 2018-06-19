from modelzoo.backend.tensor.CropLoss import CropLoss
from modelzoo.models.Net import Net
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.cropnet.CropNetDecoder import CropNetDecoder
from modelzoo.models.cropnet.CropNetEncoder import CropNetEncoder
from utils.imageprocessing.transform.ImgTransform import ImgTransform


class CropNet(Predictor):
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def __init__(self, net: Net, augmenter: ImgTransform,
                 input_shape=(52, 52), output_shape=(13, 13), color_format='yuv'):
        self._input_shape = input_shape
        self._output_shape = output_shape
        encoder = CropNetEncoder(output_shape, input_shape)
        decoder = CropNetDecoder(output_shape, input_shape)
        preprocessor = Preprocessor(augmenter, encoder, 1, input_shape, color_format)
        postprocessor = Postprocessor(decoder)
        super().__init__(preprocessor, postprocessor, net, CropLoss(), encoder, decoder)
