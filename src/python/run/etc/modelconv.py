from modelzoo.backend.tensor.ModelConverter import ModelConverter
from modelzoo.models.gatenet.GateNet import GateNet
from utils.workdir import cd_work

cd_work()
model_dir = 'out/gatev8_mixed/'
model = GateNet.v8(weight_file=model_dir + 'model.h5')

ModelConverter.convert(model=model, path=model_dir, filename='model')
