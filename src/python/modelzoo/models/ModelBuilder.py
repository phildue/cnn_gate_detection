from modelzoo.backend.tensor.simple_tf import simple_tf
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo


class ModelBuilder:

    @staticmethod
    def build(name, batch_size=8, src_dir=None):

        if src_dir is not None:
            weight_file = src_dir + 'model.h5'
        else:
            weight_file = None

        if name == "tiny_yolo":
            model = Yolo.tiny_yolo(batch_size=batch_size, weight_file=weight_file)
        elif name == "yolo":
            model = Yolo.yolo_v2(batch_size=batch_size, weight_file=weight_file)
        elif name == "thin_yolo":
            model = Yolo.thin_yolo(batch_size=batch_size, weight_file=weight_file)
        elif 'Gate' in name:
            model = GateNet.create(name, batch_size=batch_size, weight_file=weight_file)
        elif name == 'test':
            model = simple_tf()
        else:
            raise ValueError("Unknown model name!")

        return model
