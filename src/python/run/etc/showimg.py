from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import annotate_bounding_box, resize, convert_color, COLOR_YUV2BGR
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, LEGEND_BOX, LEGEND_TEXT, LEGEND_POSITION, \
    LEGEND_CORNERS
from utils.workdir import cd_work

cd_work()


def show_voc():
    dataset = VocGenerator(batch_size=2, classes=['cat']).generate()
    for batch in dataset:
        for img, label, _ in batch:
            # img, label = resize(img, shape=(416, 416), label=label)
            show(img, 'resized', labels=label)


def show_img(path):
    gate_generator = GateGenerator(path, 8, color_format='bgr', shuffle=False, label_format='xml', img_format='jpg',
                                   start_idx=0)

    for batch in gate_generator.generate():
        for img, label, _ in batch:
            img, label = resize(img, (52, 52), label=label)
            show(img, 'img', labels=label, legend=LEGEND_POSITION, thickness=1)


def show_shot(path="samplegen/resource/shots/stream/"):
    shots, labels = ShotLoad(shot_path=path, img_format='bmp').get_shots(
        n_shots=200)
    for i, img in enumerate(shots):
        ann_img, label = resize(img, (500, 500), label=labels[i])
        show(ann_img, 'labeled', labels=label)


# show_shot(path="samplegen/resource/ext/samples/bebop_merge/")
show_img(path=['resource/ext/samples/crop20k'])
# show_voc()
