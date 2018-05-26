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
        elif name == "gatev5":
            model = GateNet.v5(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev6":
            model = GateNet.v6(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev7":
            model = GateNet.v7(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev8":
            model = GateNet.v8(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev9":
            model = GateNet.v9(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev10":
            model = GateNet.v10(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev11":
            model = GateNet.v11(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev12":
            model = GateNet.v12(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev13":
            model = GateNet.v13(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev14":
            model = GateNet.v14(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev15":
            model = GateNet.v15(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev16":
            model = GateNet.v16(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev17":
            model = GateNet.v17(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev18":
            model = GateNet.v18(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev19":
            model = GateNet.v19(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev20":
            model = GateNet.v20(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev21":
            model = GateNet.v21(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev22":
            model = GateNet.v22(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev23":
            model = GateNet.v23(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev24":
            model = GateNet.v24(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev25":
            model = GateNet.v25(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev26":
            model = GateNet.v26(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev28":
            model = GateNet.v28(batch_size=batch_size, weight_file=weight_file)
        elif name == "gatev29":
            model = GateNet.v29(batch_size=batch_size, weight_file=weight_file)
        else:
            raise ValueError("Unknown model name!")

        return model
