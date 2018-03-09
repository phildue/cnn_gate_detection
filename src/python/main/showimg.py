from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import annotate_bounding_box, resize, convert_color, COLOR_YUV2BGR
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, LEGEND_BOX
from utils.workdir import work_dir

work_dir()


def show_voc():
    dataset = VocGenerator(batch_size=2).generate()
    for batch in dataset:
        for img, label, _ in batch:
            img, label = resize(img, shape=(150, 315), label=label)
            # img, label = resize(img, shape=(416, 416), label=label)
            show(img, 'resized', labels=label)


def show_img(path):
    gate_generator = GateGenerator(path, 8, color_format='bgr', shuffle=False)

    for batch in gate_generator.generate():
        for img, label, _ in batch:
            if img.format == 'yuv':
                img = convert_color(img, COLOR_YUV2BGR)
            show(img, 'labeled', labels=label, colors=[COLOR_GREEN], legend=LEGEND_BOX)

            img, label = resize(img, shape=(150, 150), label=label)
            img, label = resize(img, shape=(416, 416), label=label)
            show(img, 'resized', labels=label)


def show_shot(path="samplegen/resource/shots/stream/"):
    shots, labels = ShotLoad(shot_path=path, img_format='bmp').get_shots(
        n_shots=200)
    for i, img in enumerate(shots):
        ann_img, label = resize(img, (500, 500), label=labels[i])
        show(ann_img, 'labeled', labels=label)


# show_shot(path="samplegen/resource/shots/mult_gate_aligned/")
# show_img(path=['resource/samples/mult_gate_aligned_test/'])
show_voc()
