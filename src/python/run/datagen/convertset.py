from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
reader = DatasetParser.get_parser('resource/ext/samples/industrial_new_test/',
                                  label_format='xml',
                                  color_format='bgr',
                                  )
writer = DatasetParser.get_parser('temp/',
                                  label_format='tf_record',
                                  color_format='bgr')
images, labels = reader.read(2)
writer.write(images, labels, 'industrial_train.record')
images, labels = writer.read(2, 'industrial_train.record')

for img in images:
    show(img)
