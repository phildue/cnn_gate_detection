from modelzoo.backend.tensor.cropnet.CropGridLoss import CropGridLoss
from modelzoo.backend.tensor.cropnet.CropNetBase import CropNetBase
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

    def __init__(self, net: CropNetBase, augmenter: ImgTransform = None,
                 input_shape=(52, 52), output_shape=(13, 13), color_format='yuv', encoding='anchor', anchor_scale=None):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.n_boxes = anchor_scale[0].shape[0]
        encoder = CropNetEncoder([net.grid], input_shape, encoding=encoding, anchor_scale=anchor_scale)
        decoder = CropNetDecoder([net.grid], input_shape, encoding=encoding)
        self.grid = [net.grid]
        preprocessor = Preprocessor(augmenter, encoder, 1, input_shape, color_format)
        postprocessor = Postprocessor(decoder)
        super().__init__(preprocessor, postprocessor, net, CropGridLoss(), encoder, decoder)
