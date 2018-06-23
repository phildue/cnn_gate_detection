from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.workdir import cd_work

cd_work()
reader = DatasetParser.get_parser('resource/ext/samples/industrial_new/',
                                  label_format='xml',
                                  color_format='bgr',
                                  image_format='jpg'
                                  )
writer = DatasetParser.get_parser('resource/ext/samples/set_yolo/',
                                  label_format='yolo',
                                  color_format='bgr')
images, labels = reader.read(100)
writer.write(images, labels)
