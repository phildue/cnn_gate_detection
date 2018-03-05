from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import save_file
from utils.timing import toc, tic


class ModelEvaluator:
    def evaluate_generator(self, generator: DatasetGenerator, n_batches=10):
        it = iter(generator.generate())
        labels_true = []
        labels_pred = []
        image_files = []
        for i in range(n_batches):
            batch = next(it)
            images = [b[0] for b in batch]
            labels = [b[1] for b in batch]
            image_files_batch = [b[2] for b in batch]

            tic()
            predictions = self.model.predict(images)
            labels_true.extend(labels)
            labels_pred.extend(predictions)
            image_files.extend(image_files_batch)
            if self.verbose:
                toc("Evaluated batch {0:d}/{1:d} in ".format(i, n_batches))

            if self.out_file is not None:
                content = {'labels_true': labels_true,
                           'labels_pred': labels_pred,
                           'image_files': image_files}
                save_file(content, self.out_file)

    def __init__(self, model: Predictor, out_file=None, verbose=True):
        self.verbose = verbose
        self.out_file = out_file
        self.model = model
