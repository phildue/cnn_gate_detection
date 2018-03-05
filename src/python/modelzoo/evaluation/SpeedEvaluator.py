from pprint import pprint

import numpy as np

from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import save_file
from utils.timing import toc, tic


class SpeedEvaluator(ModelEvaluator):
    def __init__(self, model: Predictor, out_file=None, verbose=True):
        super().__init__(model, out_file, verbose)

    def evaluate_generator(self, generator: DatasetGenerator, n_batches=10):
        it = iter(generator.generate())
        results = []
        for i in range(n_batches):
            batch = next(it)
            images = [b[0] for b in batch]
            tic()
            sample_t = []
            for s in images:
                sample_t.append(self.model.preprocessor.encode_img(s.yuv))
            sample_t = np.concatenate(sample_t, 0)
            time_enc = toc()
            tic()
            netout = self.model.net.predict(sample_t)
            time_pred = toc()

            tic()
            predictions = []
            for i in range(netout.shape[0]):
                predictions.append(self.model.postprocessor.postprocess(netout[i]))
            time_pp = toc()
            fps_enc = len(images) / time_enc
            fps_pred = len(images) / time_pred
            fps_pp = len(images) / time_pp

            fps_total = len(images) / (time_enc + time_pred + time_pp)

            results_batch = {'fps_enc': fps_enc,
                             'fps_pred': fps_pred,
                             'fps_pp': fps_pp,
                             'fps_total': fps_total}

            results.append(results_batch)

            if self.verbose:
                pprint("Evaluated batch {0:d}/{1:d}".format(i, n_batches))
                pprint(results_batch)
            if self.out_file is not None:
                content = {'results': results}
                save_file(content, self.out_file)

        return results
