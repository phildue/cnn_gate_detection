from utils.fileaccess.labelparser.PklParser import PklParser
from utils.fileaccess.labelparser.XmlParser import XmlParser


class DatasetParser:

    @staticmethod
    def get_parser(directory: str, label_format, color_format, start_idx=0, image_format='jpg'):
        if label_format is 'xml':
            return XmlParser(directory, color_format, start_idx, image_format)
       # elif label_format is 'tf_record':
       #     return TfRecordParser(directory, color_format, start_idx, image_format)
        elif label_format is 'pkl':
            return PklParser(directory, color_format, start_idx, image_format)
        else:
            print("Label format not known!!")

    @staticmethod
    def read_label(label_format, filepath):
        if label_format is 'xml':
            return XmlParser.read_label(filepath)
        #elif label_format is 'tf_record':
        #    return TfRecordParser.read_label(filepath)
        elif label_format is 'pkl':
            return PklParser.read_label(filepath)
        else:
            print("Label format not known!!")
