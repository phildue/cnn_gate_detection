from modelzoo.backend.tensor.CropGridLoss import CropGridLoss
from modelzoo.backend.tensor.cropnet.CropNet2L import CropNet2L
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

    def __init__(self, net: CropNet2L, augmenter: ImgTransform = None,
                 input_shape=(52, 52), output_shape=(13, 13), color_format='yuv'):
        self._input_shape = input_shape
        self._output_shape = output_shape
        encoder = CropNetEncoder(net.grid, input_shape, encoding='grid')
        decoder = CropNetDecoder(net.grid, input_shape)
        preprocessor = Preprocessor(augmenter, encoder, 1, input_shape, color_format)
        postprocessor = Postprocessor(decoder)
        super().__init__(preprocessor, postprocessor, net, CropGridLoss(), encoder, decoder)
