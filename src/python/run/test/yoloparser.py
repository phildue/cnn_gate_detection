from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
parser = DatasetParser.get_parser(label_format='yolo',
                                  color_format='bgr',
                                  directory='resource/ext/samples/yolo_test/samples',
                                  img_norm=(208, 208))

images, labels = parser.read(10)

for img, label in zip(images, labels):
    show(img, labels=label)
