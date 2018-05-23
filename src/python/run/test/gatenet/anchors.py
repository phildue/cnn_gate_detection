from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.workdir import cd_work

cd_work()
batch_size = 1
encoder = GateNetEncoder()
anchors = encoder._generate_anchors()
print(anchors)
dataset = GateGenerator(["resource/ext/samples/industrial_new/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=batch_size).generate()
batch = next(dataset)

batch = [resize(b[0], (416, 416), label=b[1]) for b in batch]

for i in range(batch_size):
    img, label = batch[i]
    print("Assigned")
    anchors_assigned = encoder._assign_true_boxes(anchors, BoundingBox.from_label(label))
    print(anchors_assigned)
    anchors_encoded = encoder._encode_coords(anchors_assigned, anchors)
    print("Encoded")
    print(anchors_encoded)
