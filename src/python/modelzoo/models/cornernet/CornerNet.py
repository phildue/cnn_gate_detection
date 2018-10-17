from markdown.preprocessors import Preprocessor

from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.cornernet import CornerNetDecoder
from modelzoo.models.cornernet import CornerNetEncoder
from modelzoo.models.cornernet import PolygonLoss
from modelzoo.models.cornernet import PolygonNet


class CornerNet(Predictor):

    def __init__(self, image_shape, n_polygon):
        decoder = CornerNetDecoder(image_shape)
        loss = PolygonLoss()
        super().__init__(Preprocessor(), Postprocessor(decoder),PolygonNet(loss), loss,CornerNetEncoder(image_shape,n_polygon), decoder)
        self.n_polygon = n_polygon
        self.image_shape = image_shape

    @property
    def input_shape(self):
        return self.image_shape

    @property
    def output_shape(self):
        return self.n_polygon * 2
