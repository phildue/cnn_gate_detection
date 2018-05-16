from modelzoo.backend.tensor.ModelConverter import ModelConverter
from utils.workdir import cd_work

cd_work()
quantize = False

ModelConverter('gatev8', 'logs/gatev8_mixed/').finalize(quantize)
