import numpy as np

from modelzoo.evaluation.Metric import Metric
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import save_file, create_dirs
from utils.imageprocessing.Imageprocessing import show
from utils.labels.utils import resize_label
from utils.timing import toc, tic
from utils.imageprocessing.Backend import resize

class ModelEvaluator:
    def evaluate_generator(self, generator: DatasetGenerator, n_batches=None):
        if n_batches is None:
            n_batches = int(np.floor(generator.n_samples / generator.batch_size))
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
            # for i,p in enumerate(predictions):
            #     img = resize(images[i],self.model.input_shape)
            #     show(img,labels=predictions[i])
            labels = [resize_label(l,images[0].shape[:2],self.model.input_shape) for l in labels]
            labels_true.extend(labels)
            labels_pred.extend(predictions)
            image_files.extend(image_files_batch)
            if self.verbose:
                toc("Evaluated batch {0:d}/{1:d} in ".format(i, n_batches))

            if self.out_file is not None:
                content = {'labels_true': labels_true,
                           'labels_pred': labels_pred,
                           'image_files': image_files}
                save_file(content, self.out_file, verbose=self.verbose)

        return labels_true, labels_pred, image_files

    def __init__(self, model: Predictor, out_file=None, verbose=True):
        self.verbose = verbose
        self.out_file = out_file
        self.model = model
