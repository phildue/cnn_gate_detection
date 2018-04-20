from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
reader = DatasetParser.get_parser('resource/ext/samples/industrial_new_test/',
                                  label_format='xml',
                                  color_format='bgr',
                                  image_format='jpg'
                                  )
writer = DatasetParser.get_parser('resource/ext/samples/industrial_new_test/',
                                  label_format='tf_record',
                                  color_format='bgr')
images, labels = reader.read(500)
writer.write(images, labels, 'industrial_valid.record')
