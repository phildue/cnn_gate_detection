from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo


class ModelFactory:

    @staticmethod
    def build(name, batch_size=8, src_dir=None, img_res=(416, 416), grid=[(13, 13)], anchors=None):

        if src_dir is not None:
            weight_file = src_dir + 'model.h5'
        else:
            weight_file = None

        if name == "tiny_yolo":
            model = Yolo.tiny_yolo(batch_size=batch_size, weight_file=weight_file, norm=img_res)
        elif name == "yolo":
            model = Yolo.yolo_v2(batch_size=batch_size, weight_file=weight_file, norm=img_res)
        elif name == "thin_yolo":
            model = Yolo.thin_yolo(batch_size=batch_size, weight_file=weight_file, norm=img_res)
        elif 'Gate' in name:
            model = GateNet.create(name, batch_size=batch_size, weight_file=weight_file, norm=img_res, grid=grid,
                                   anchors=anchors)
        elif name == 'test':
            model = simple_tf()
        else:
            raise ValueError("Unknown model name!")

        return model
