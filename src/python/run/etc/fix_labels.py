from utils.fileaccess.labelparser.YoloParser import YoloParser
from utils.workdir import cd_work

cd_work()
dir = 'resource/ext/samples/yolo_new_'
images, labels = YoloParser(dir,color_format='bgr').read()
for l in labels:
    YoloParser.write_label(label,)