import glob

from imageprocessing.Backend import imwrite, imread

from src.python.utils.imageprocessing.Image import Image


def write_imgs(path: str, samples: [Image], file_format='jpg'):
    return ImgFileParser(path, file_format).write(samples)


def read_imgs(path: str, file_format='jpg'):
    return ImgFileParser(path, file_format).read()


class ImgFileParser:
    def __init__(self, directory: str, file_format: str, data_format: str = None, start_idx=0):
        self.data_format = data_format
        self.file_format = file_format
        self.directory = directory
        self.offset = start_idx
        self.files = sorted(glob.glob(self.directory + "/*." + self.file_format))

    def write(self, samples: [Image]):
        for s in samples:
            imwrite(s, '{0}/{1:05d}.{2}'.format(self.directory, self.offset, self.file_format))
            self.offset += 1

    def read_batch(self, batch_size) -> [[Image]]:
        files_batch = []
        for i, f in self.files:
            files_batch.append(f)
            if len(files_batch) >= batch_size:
                yield self.__read(files_batch)
                files_batch = []

    def read(self, n=0) -> [Image]:
        return self.__read(self.files, n)

    @staticmethod
    def __read(paths, n=0) -> [Image]:
        # Scan through files, read every image and respective label
        imgs = []
        for i, f in enumerate(paths):
            if 0 < n < i: break
            imgs.append(imread(f))

        return imgs
